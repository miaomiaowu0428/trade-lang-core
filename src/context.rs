//! 任务执行上下文
//!
//! `TradeTaskContext` 是跨所有并发 task 共享的状态载体：
//!   - `vars`   — 策略变量表，parking_lot RwLock 保护，同步读写无异步调度开销
//!   - `cancel` — Done 信号 token：任意位置触发后整个 pipeline 取消进入 finally
//!   - `start`  — 策略入场时间戳，供 Timeout 等计时插件使用

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use tokio::sync::{Mutex, Notify};
use tokio_util::sync::CancellationToken;
use trade_meta_compiler::RuntimeValue;

use crate::ConfirmHandle;

/// 任务上下文（所有并发 task 持有同一个 Arc<TradeTaskContext>）
#[derive(Clone)]
pub struct TradeTaskContext {
    /// 策略变量表（对应 trade 文件 vars 块中声明的变量）
    pub vars: Arc<RwLock<HashMap<String, RuntimeValue>>>,
    /// Done 取消信号：`signal_done()` 触发后所有持有该 token 的 task 应及时退出
    pub cancel: CancellationToken,
    /// 策略执行开始时间
    pub start: Instant,
    /// 隐式上下文存储：Vec<(protocol_name, value)>，条目数极少，线性扫描最优
    pub contexts: Arc<RwLock<Vec<(&'static str, Arc<dyn Any + Send + Sync>)>>>,
    /// condition 推入的待 confirm/cancel 的 handle 列表
    confirm_handles: Arc<Mutex<Vec<Box<dyn ConfirmHandle>>>>,
    /// buy 阶段是否已成功完成（sell 阶段 push 的 handle 直接 confirm）
    buy_confirmed: Arc<std::sync::atomic::AtomicBool>,
    /// buy_confirmed 的变更通知。`confirm_all_handles` 调用后 notify_waiters，
    /// 等待方使用 `wait_buy_confirmed().await` 阻塞至 buy 成功。
    buy_confirmed_notify: Arc<Notify>,
}

impl TradeTaskContext {
    pub fn new() -> Self {
        Self {
            vars: Arc::new(RwLock::new(HashMap::new())),
            cancel: CancellationToken::new(),
            start: Instant::now(),
            contexts: Arc::new(RwLock::new(Vec::new())),
            confirm_handles: Arc::new(Mutex::new(Vec::new())),
            buy_confirmed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            buy_confirmed_notify: Arc::new(Notify::new()),
        }
    }

    /// 创建绑定到父 cancel token 的上下文
    ///
    /// 父 token 取消时，该上下文的 cancel 也会被取消（级联）。
    /// 但 `signal_done()` 只取消本上下文，不影响父 token。
    pub fn with_parent_cancel(parent: &CancellationToken) -> Self {
        Self {
            vars: Arc::new(RwLock::new(HashMap::new())),
            cancel: parent.child_token(),
            start: Instant::now(),
            contexts: Arc::new(RwLock::new(Vec::new())),
            confirm_handles: Arc::new(Mutex::new(Vec::new())),
            buy_confirmed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            buy_confirmed_notify: Arc::new(Notify::new()),
        }
    }

    /// 创建 Spawn 后台任务专用的子上下文。
    ///
    /// - `vars` / `contexts` / `buy_confirmed` 与父共享（Spawn 需要读 pump_token、trade_hint 等）
    /// - `cancel` 是父 token 的 child token：父取消 → 子也取消（级联向下）
    ///   但子调 `signal_done()` **不会**影响父 pipeline（隔离向上）
    /// - `confirm_handles` 全新空列表（Spawn 任务不参与 buy 确认流程）
    pub fn spawn_child(parent: &Arc<TradeTaskContext>) -> Self {
        Self {
            vars: Arc::clone(&parent.vars),
            cancel: parent.cancel.child_token(),
            start: parent.start,
            contexts: Arc::clone(&parent.contexts),
            confirm_handles: Arc::new(Mutex::new(Vec::new())),
            buy_confirmed: Arc::clone(&parent.buy_confirmed),
            buy_confirmed_notify: Arc::clone(&parent.buy_confirmed_notify),
        }
    }

    // ── 变量访问 ──────────────────────────────────────────────────────────

    /// Sync variant — 用于 pipeline 内部热路径，避免 async 状态机开销。
    #[inline]
    pub fn get_var_sync(&self, name: &str) -> Option<RuntimeValue> {
        self.vars.read().get(name).cloned()
    }

    /// Sync variant — 用于 pipeline 内部热路径。
    #[inline]
    pub fn set_var_sync(&self, name: &str, value: RuntimeValue) {
        self.vars.write().insert(name.to_string(), value);
    }

    /// Sync variant — 返回当前全量快照。
    #[inline]
    pub fn snapshot_vars_sync(&self) -> HashMap<String, RuntimeValue> {
        self.vars.read().clone()
    }

    // async 包装，保持与 macro 生成代码及外部 handler 的兼容性
    #[inline]
    pub async fn get_var(&self, name: &str) -> Option<RuntimeValue> {
        self.get_var_sync(name)
    }

    #[inline]
    pub async fn set_var(&self, name: &str, value: RuntimeValue) {
        self.set_var_sync(name, value);
    }

    #[inline]
    pub async fn snapshot_vars(&self) -> HashMap<String, RuntimeValue> {
        self.snapshot_vars_sync()
    }

    // ── 隐式上下文 ──────────────────────────────────────────────────────────

    /// Sync variant
    #[inline]
    pub fn produce_context_sync<T: Any + Send + Sync>(&self, protocol: &'static str, value: T) {
        self.contexts.write().push((protocol, Arc::new(value)));
    }

    /// Sync variant
    #[inline]
    pub fn get_context_sync<T: Any + Send + Sync>(&self, protocol: &'static str) -> Option<Arc<T>> {
        let guard = self.contexts.read();
        guard
            .iter()
            .find(|(k, _)| *k == protocol)
            .and_then(|(_, v)| Arc::clone(v).downcast::<T>().ok())
    }

    /// Sync variant — 取出并移除上下文条目。
    #[inline]
    pub fn consume_context_sync<T: Any + Send + Sync>(
        &self,
        protocol: &'static str,
    ) -> Option<Arc<T>> {
        let mut guard = self.contexts.write();
        let pos = guard.iter().position(|(k, _)| *k == protocol)?;
        let (_, v) = guard.swap_remove(pos);
        v.downcast::<T>().ok()
    }

    /// Sync variant
    #[inline]
    pub fn has_context_sync(&self, protocol: &'static str) -> bool {
        self.contexts.read().iter().any(|(k, _)| *k == protocol)
    }

    // async 包装
    #[inline]
    pub async fn produce_context<T: Any + Send + Sync>(&self, protocol: &'static str, value: T) {
        self.produce_context_sync(protocol, value);
    }

    #[inline]
    pub async fn get_context<T: Any + Send + Sync>(
        &self,
        protocol: &'static str,
    ) -> Option<Arc<T>> {
        self.get_context_sync(protocol)
    }

    #[inline]
    pub async fn consume_context<T: Any + Send + Sync>(
        &self,
        protocol: &'static str,
    ) -> Option<Arc<T>> {
        self.consume_context_sync(protocol)
    }

    #[inline]
    pub async fn has_context(&self, protocol: &'static str) -> bool {
        self.has_context_sync(protocol)
    }

    // ── Confirm Handle ────────────────────────────────────────────────────

    /// condition evaluate 时调用：推入一个 handle。
    /// 若 buy 已成功（sell 阶段），直接 confirm。
    pub async fn push_confirm_handle(&self, handle: Box<dyn ConfirmHandle>, self_arc: &Arc<Self>) {
        if self
            .buy_confirmed
            .load(std::sync::atomic::Ordering::Acquire)
        {
            // 已过 buy 阶段，直接 confirm
            handle.confirm(self_arc).await;
        } else {
            self.confirm_handles.lock().await.push(handle);
        }
    }

    /// buy 成功后调用：confirm 所有已推入的 handle，并标记 buy_confirmed
    pub async fn confirm_all_handles(self: &Arc<Self>) {
        self.buy_confirmed
            .store(true, std::sync::atomic::Ordering::Release);
        // 唤醒所有 wait_buy_confirmed 的等待者（如 Spawn 中的 condition）
        self.buy_confirmed_notify.notify_waiters();
        let handles: Vec<_> = self.confirm_handles.lock().await.drain(..).collect();
        for h in handles {
            h.confirm(self).await;
        }
    }

    /// buy 失败后调用：cancel 所有已推入的 handle
    pub async fn cancel_all_handles(&self) {
        let handles: Vec<_> = self.confirm_handles.lock().await.drain(..).collect();
        for h in handles {
            h.cancel().await;
        }
    }

    // ── Done 信号 ─────────────────────────────────────────────────────────

    pub fn signal_done(&self) {
        self.cancel.cancel();
    }

    pub fn is_done(&self) -> bool {
        self.cancel.is_cancelled()
    }

    pub fn done_future(&self) -> tokio_util::sync::WaitForCancellationFuture<'_> {
        self.cancel.cancelled()
    }
    // ── buy_confirmed 门──────────────────────────────────────────

    /// 查询 buy 是否已确认（sell 阶段 后恒为 true）
    pub fn is_buy_confirmed(&self) -> bool {
        self.buy_confirmed
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// 阻塞至 buy 确认。若已确认立即返回；否则等待 `confirm_all_handles` 唤醒。
    /// 主要面向在 buy 块 Spawn 中推入的 condition：为避免漏消息早注册订阅，
    /// 但不希望在 buy 确认前就触发下游逻辑。
    pub async fn wait_buy_confirmed(&self) {
        if self.is_buy_confirmed() {
            return;
        }
        // 先注册 notified（避免 notify_waiters 与下面的 load 之间的竞态）
        let notified = self.buy_confirmed_notify.notified();
        tokio::pin!(notified);
        // 二次检查：若在注册前已 confirm，直接返回
        if self.is_buy_confirmed() {
            return;
        }
        notified.await;
    }
    pub fn child_token(&self) -> CancellationToken {
        self.cancel.child_token()
    }
}

impl Default for TradeTaskContext {
    fn default() -> Self {
        Self::new()
    }
}
