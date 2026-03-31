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

use async_trait::async_trait;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use trade_meta_compiler::ast::{Condition, ExecutorItem};
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
}

impl MonitorMessage {
    /// 创建携带单个上下文的触发消息
    pub fn single(protocol: &'static str, value: impl Any + Send + Sync) -> Self {
        Self {
            contexts: vec![(protocol, Arc::new(value))],
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
#[async_trait]
pub trait ConditionHandler: Send + Sync {
    async fn evaluate(
        &self,
        args: &HashMap<String, RuntimeValue>,
        ctx: &Arc<TradeTaskContext>,
    ) -> bool;
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

// ── Pipeline 操作抽象 & 控制流 Handler ────────────────────────────────────────

/// Pipeline 操作接口抽象
///
/// 控制流 handler（BranchedCallHandler / AllCallHandler）通过此接口
/// 调用 pipeline 的条件评估与执行器执行能力，而不直接依赖具体的 TradePipeline。
#[async_trait]
pub trait PipelineOps: Send + Sync + 'static {
    /// 评估条件
    async fn eval_condition(&self, cond: &Condition) -> bool;
    /// 执行一组 ExecutorItem，返回 true 表示触发了 Done
    async fn exec_executor_items(&self, items: &[ExecutorItem]) -> bool;
    /// 当前任务是否已 Done
    fn is_done(&self) -> bool;
    /// 发出 Done 信号
    fn signal_done(&self);
    /// 克隆一份 ops 句柄（用于 Spawn 等需要 move 进新 task 的场景）
    fn clone_ops(&self) -> Arc<dyn PipelineOps>;
}

/// 分支调用 handler（Spawn / OneOf / 自定义分支语句）
///
/// 每个分支由 (Condition, Vec\<ExecutorItem\>) 组成。
/// handler 决定如何调度这些分支（并发竞争、后台派生、顺序匹配等）。
#[async_trait]
pub trait BranchedCallHandler: Send + Sync {
    async fn execute(
        &self,
        branches: &[(Condition, Vec<ExecutorItem>)],
        ops: Arc<dyn PipelineOps>,
    ) -> bool;
}

/// All 调用 handler（并发评估所有条件，全部满足后执行）
#[async_trait]
pub trait AllCallHandler: Send + Sync {
    async fn execute(
        &self,
        conditions: &[Condition],
        executors: &[ExecutorItem],
        ops: Arc<dyn PipelineOps>,
    ) -> bool;
}

// ── RuntimeRegistry ───────────────────────────────────────────────────────────

/// 运行时注册表：将符号名映射到其运行时 handler 实现
///
/// 由 impl crate 填充（如 mock_dex 或 solana-impl），由 runtime provider 读取调度。
#[derive(Default)]
pub struct RuntimeRegistry {
    pub data_items: HashMap<String, Arc<dyn DataItemHandler>>,
    pub executors: HashMap<String, Arc<dyn ExecutorHandler>>,
    pub conditions: HashMap<String, Arc<dyn ConditionHandler>>,
    pub monitors: HashMap<String, Arc<dyn MonitorHandler>>,
    pub branched_calls: HashMap<String, Arc<dyn BranchedCallHandler>>,
    pub all_calls: HashMap<String, Arc<dyn AllCallHandler>>,
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

    pub fn register_branched_call(&mut self, name: &str, handler: Arc<dyn BranchedCallHandler>) {
        self.branched_calls.insert(name.to_string(), handler);
    }

    pub fn register_all_call(&mut self, name: &str, handler: Arc<dyn AllCallHandler>) {
        self.all_calls.insert(name.to_string(), handler);
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
