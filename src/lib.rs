//! Trade-Lang Core
//!
//! 定义策略框架的核心抽象接口，所有 runtime provider 和 handler 实现共享这些定义。
//!
//! 包含：
//!   - Handler traits：MonitorHandler, ExecutorHandler, DataItemHandler, ConditionHandler
//!   - RuntimeRegistry：符号名 → handler 映射表
//!   - TradeTaskContext：任务执行上下文（变量、隐式上下文、Done 信号）
//!   - MonitorMessage：Monitor 触发消息
//!
//! 依赖关系：
//!   trade-meta-compiler (TypeSpec, RuntimeValue, SymbolMetadata, SymbolRegistry)
//!         ↑
//!   trade-lang-core  ← 本 crate
//!         ↑                       ↑
//!   runtime-provider          solana-impl

use ahash::AHashMap;
use ahash::AHashMap as HashMap;
use async_trait::async_trait;
use std::any::Any;
use std::sync::Arc;

use trade_meta_compiler::{RuntimeValue, SymbolCategory, SymbolRegistry, TypeSpec};

pub mod context;
pub use context::TradeTaskContext;

// ── Re-exports for macro-generated code ───────────────────────────────────────
pub use tokio::sync::mpsc as monitor_mpsc;
pub use tokio_util::sync::CancellationToken;

/// Executor handler 返回类型：成功值 or 错误
pub type ExecutorResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Spawn a task on the tokio runtime (re-export for macro-generated adapter code)
pub fn spawn_task<F>(f: F) -> tokio::task::JoinHandle<F::Output>
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    tokio::spawn(f)
}

// ── MonitorMessage ────────────────────────────────────────────────────────────

/// Monitor 触发时发出的消息，携带需要注入到新 TradeTaskContext 的上下文数据
pub struct MonitorMessage {
    /// (protocol_name, context_value) 列表
    pub contexts: Vec<(&'static str, Arc<dyn Any + Send + Sync>)>,
    /// 触发信号的源头时刻（如 shred 收到的 Instant），由 Monitor 填写。
    /// Runner 创建 ctx 时会把它复制到 `TradeTaskContext::sig_time`，
    /// pipeline 自动用其打印每个 Symbol 的端到端耗时。
    pub sig_time: Option<std::time::Instant>,
}

impl MonitorMessage {
    /// 创建携带单个上下文的触发消息
    pub fn single(protocol: &'static str, value: impl Any + Send + Sync) -> Self {
        Self {
            contexts: vec![(protocol, Arc::new(value))],
            sig_time: None,
        }
    }
}

// ── Handler Traits ────────────────────────────────────────────────────────────

/// DataItem：读取链上/环境数据（通常即时返回）
#[async_trait]
pub trait DataItemHandler: Send + Sync {
    fn declared_return_type(&self) -> TypeSpec;
    async fn get(
        &self,
        args: &HashMap<String, RuntimeValue>,
        ctx: &Arc<TradeTaskContext>,
    ) -> RuntimeValue;
}

/// Executor：执行一个操作（交易等），可选返回值
#[async_trait]
pub trait ExecutorHandler: Send + Sync {
    fn declared_return_type(&self) -> Option<TypeSpec>;
    async fn execute(
        &self,
        args: &HashMap<String, RuntimeValue>,
        ctx: &Arc<TradeTaskContext>,
    ) -> Option<RuntimeValue>;
}

/// Condition：评估条件是否满足（可阻塞轮询直到条件成立）
///
/// **注意**：Condition 需要持续轮询时，应通过 `ctx.done_future()` 或
/// `ctx.cancel.cancelled()` 来响应 Done 信号并提前退出。
///
/// 返回值：`(triggered, side_value)`
///   - `triggered`  — 条件是否满足
///   - `side_value` — 可选的偏值（如累计询价次数），供 `let x = Cond(...)` 捕获
#[async_trait]
pub trait ConditionHandler: Send + Sync {
    async fn evaluate(
        &self,
        args: &HashMap<String, RuntimeValue>,
        ctx: &Arc<TradeTaskContext>,
    ) -> (bool, Option<RuntimeValue>);
}

// ── ConfirmHandle ─────────────────────────────────────────────────────────

/// 需要 confirm/cancel 生命周期管理的句柄。
///
/// condition 在 evaluate 时可向 ctx 推入 handle：
///   - buy 阶段结束且成功 → pipeline 调 `confirm` 使其生效
///   - buy 失败 → pipeline 调 `cancel` 清理资源
///   - sell 阶段中推入的 handle → 立即 confirm（已过 buy 阶段）
#[async_trait]
pub trait ConfirmHandle: Send + Sync {
    async fn confirm(&self, ctx: &Arc<TradeTaskContext>);
    async fn cancel(&self);
}

/// Monitor：启动链上事件监听，通过 channel 发送触发消息
///
/// 与其他 Handler 不同，Monitor 独立于交易流程运行：
/// - `start()` 启动监听并返回消息接收端
/// - 每条消息代表一次触发，Runner 为每条消息创建独立的交易流程
/// - 当 cancel token 被触发时，Monitor 应停止监听并关闭 channel
#[async_trait]
pub trait MonitorHandler: Send + Sync {
    async fn start(
        &self,
        args: &HashMap<String, RuntimeValue>,
        cancel: CancellationToken,
    ) -> monitor_mpsc::Receiver<MonitorMessage>;
}

// ── RuntimeRegistry ───────────────────────────────────────────────────────────

/// 运行时注册表：将符号名映射到其运行时 handler 实现
///
/// 由 impl crate 填充（如 mock_dex 或 solana-impl），由 runtime provider 读取调度。
/// 使用 `ahash::AHashMap` 替代标准库 HashMap，string 键查找速度约快 2-3×。
pub struct RuntimeRegistry {
    pub data_items: AHashMap<String, Arc<dyn DataItemHandler>>,
    pub executors: AHashMap<String, Arc<dyn ExecutorHandler>>,
    pub conditions: AHashMap<String, Arc<dyn ConditionHandler>>,
    pub monitors: AHashMap<String, Arc<dyn MonitorHandler>>,
}

impl Default for RuntimeRegistry {
    fn default() -> Self {
        Self {
            data_items: AHashMap::new(),
            executors: AHashMap::new(),
            conditions: AHashMap::new(),
            monitors: AHashMap::new(),
        }
    }
}

impl RuntimeRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_data_item(&mut self, name: &str, handler: Arc<dyn DataItemHandler>) {
        self.data_items.insert(name.to_string(), handler);
    }

    pub fn register_executor(&mut self, name: &str, handler: Arc<dyn ExecutorHandler>) {
        self.executors.insert(name.to_string(), handler);
    }

    pub fn register_condition(&mut self, name: &str, handler: Arc<dyn ConditionHandler>) {
        self.conditions.insert(name.to_string(), handler);
    }

    pub fn register_monitor(&mut self, name: &str, handler: Arc<dyn MonitorHandler>) {
        self.monitors.insert(name.to_string(), handler);
    }

    /// 验证运行时注册表与符号表的一致性
    pub fn validate_against(&self, symbol_registry: &SymbolRegistry) -> Result<(), Vec<String>> {
        let mut errors: Vec<String> = vec![];

        // ── DataItem ──────────────────────────────────────────────────────────
        for meta in symbol_registry.all_symbols(SymbolCategory::DataItem) {
            if !self.data_items.contains_key(meta.name) {
                errors.push(format!(
                    "DataItem '{}' defined in symbol table but not implemented",
                    meta.name
                ));
            }
        }
        for (name, handler) in &self.data_items {
            match symbol_registry.lookup(name, SymbolCategory::DataItem) {
                None => errors.push(format!(
                    "DataItem '{}' implemented but not defined in symbol table",
                    name
                )),
                Some(meta) => {
                    let declared = handler.declared_return_type();
                    if meta.returns.as_ref() != Some(&declared) {
                        errors.push(format!(
                            "DataItem '{}': return type mismatch — impl declares {:?}, symbol table says {:?}",
                            name, declared, meta.returns
                        ));
                    }
                }
            }
        }

        // ── Executor ──────────────────────────────────────────────────────────
        for meta in symbol_registry.all_symbols(SymbolCategory::Executor) {
            if !self.executors.contains_key(meta.name) {
                errors.push(format!(
                    "Executor '{}' defined in symbol table but not implemented",
                    meta.name
                ));
            }
        }
        for (name, handler) in &self.executors {
            match symbol_registry.lookup(name, SymbolCategory::Executor) {
                None => errors.push(format!(
                    "Executor '{}' implemented but not defined in symbol table",
                    name
                )),
                Some(meta) => {
                    let declared = handler.declared_return_type();
                    if declared != meta.returns {
                        errors.push(format!(
                            "Executor '{}': return type mismatch — impl declares {:?}, symbol table says {:?}",
                            name, declared, meta.returns
                        ));
                    }
                }
            }
        }

        // ── Condition ─────────────────────────────────────────────────────────
        for meta in symbol_registry.all_symbols(SymbolCategory::Condition) {
            if !self.conditions.contains_key(meta.name) {
                errors.push(format!(
                    "Condition '{}' defined in symbol table but not implemented",
                    meta.name
                ));
            }
        }
        for name in self.conditions.keys() {
            if symbol_registry
                .lookup(name, SymbolCategory::Condition)
                .is_none()
            {
                errors.push(format!(
                    "Condition '{}' implemented but not defined in symbol table",
                    name
                ));
            }
        }

        // ── Monitor ───────────────────────────────────────────────────────────
        for meta in symbol_registry.all_symbols(SymbolCategory::Monitor) {
            if !self.monitors.contains_key(meta.name) {
                errors.push(format!(
                    "Monitor '{}' defined in symbol table but not implemented",
                    meta.name
                ));
            }
        }
        for name in self.monitors.keys() {
            if symbol_registry
                .lookup(name, SymbolCategory::Monitor)
                .is_none()
            {
                errors.push(format!(
                    "Monitor '{}' implemented but not defined in symbol table",
                    name
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}
