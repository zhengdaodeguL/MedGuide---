import { FormEvent, KeyboardEvent as ReactKeyboardEvent, useEffect, useRef, useState } from "react";
import { PASSWORD_INPUT_MAX_LENGTH, USERNAME_INPUT_MAX_LENGTH, validateAuthCredentials } from "./auth-validation";
import { safeCitationSourceUrl } from "./source-url";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  BookOpen,
  Check,
  Clock3,
  Eye,
  EyeOff,
  HeartPulse,
  Info,
  LayoutDashboard,
  LockKeyhole,
  LogIn,
  LogOut,
  MessageCircle,
  PanelRight,
  Plus,
  Send,
  ShieldCheck,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  UserPlus,
  UserRound,
  X,
} from "lucide-react";

type RiskLevel = "low" | "watch" | "high";
type DisplayRiskLevel = RiskLevel | "unknown";
type SessionStatus = "idle" | "creating" | "ready" | "error";
type AuthStatus = "checking" | "anonymous" | "authenticated";
type AuthMode = "login" | "register";

type AuthUser = {
  username: string;
};

type Citation = {
  id: string;
  title: string;
  category: string;
  source: string;
  updated_at: string;
  snippet: string;
  score: number;
  retrieval: string;
  source_url?: string;
};

type SessionState = {
  session_id: string;
  turn_count: number;
  profile: Record<string, string | number | string[]>;
  risk_level?: RiskLevel | null;
  risk_flags: string[];
  intent: string;
  next_question: string | null;
  mode: string;
  /** Optional capability fields supplied by newer API versions. */
  environment?: string;
  network_enabled?: boolean;
  risk_assessed?: boolean;
  workflow_engine?: string;
  summary?: string;
};

type StructuredResult = {
  intent: string;
  sql: string;
  columns: string[];
  rows: Record<string, string | number>[];
  blocked: boolean;
  reason?: string | null;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  sessionId?: string;
  citations?: Citation[];
  structuredResult?: StructuredResult | null;
  latency?: number;
  requestId?: string;
};

type PipelineEvent = { node: string; label: string; elapsed_ms: number };

type ChatResponseBody = {
  session_id: string;
  response_id?: string;
  answer: string;
  state: SessionState;
  citations: Citation[];
  structured_result?: StructuredResult | null;
  latency_ms?: number;
  summary?: string;
};

function apiHeaders(init?: HeadersInit) {
  return new Headers(init);
}

function apiFetch(path: string, init: RequestInit = {}) {
  return fetch(path, { ...init, credentials: "include" });
}

function createRequestId(prefix: string) {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

class HttpResponseError extends Error {
  readonly status: number;
  readonly detail?: string;

  constructor(status: number, detail?: string) {
    super(`HTTP ${status}`);
    this.name = "HttpResponseError";
    this.status = status;
    this.detail = detail;
  }
}

const initialMessage: Message = {
  id: "welcome",
  role: "assistant",
  timestamp: Date.now(),
  content:
    "你好，我是 MedGuide。可以帮你整理健康信息、判断就医紧迫性、推荐就诊方向，并解释常见药品与检查注意事项。先说说：现在最困扰你的症状是什么？",
};

function formatTime(date = new Date()) {
  return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}

function formatDateDivider(timestamp: number) {
  const date = new Date(timestamp);
  const today = new Date();
  const sameDay = date.getFullYear() === today.getFullYear()
    && date.getMonth() === today.getMonth()
    && date.getDate() === today.getDate();
  const label = sameDay ? "今天" : date.toLocaleDateString("zh-CN", { month: "long", day: "numeric" });
  return `${label} ${formatTime(date)}`;
}

function isAbortError(error: unknown) {
  return typeof error === "object" && error !== null && "name" in error && error.name === "AbortError";
}

async function ensureResponseOk(response: Response) {
  if (response.ok) return;
  let detail: string | undefined;
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") {
      detail = body.detail;
    } else if (Array.isArray(body.detail)) {
      const messages = body.detail
        .map((item) => (typeof item === "object" && item !== null && "msg" in item ? item.msg : null))
        .filter((item): item is string => typeof item === "string");
      detail = messages.length ? messages.join("；") : undefined;
    }
  } catch {
    // Some proxies return an empty or non-JSON error body.
  }
  throw new HttpResponseError(response.status, detail);
}

function formatHttpError(error: HttpResponseError) {
  switch (error.status) {
    case 400:
    case 422:
      return "输入未通过校验，请确认描述不为空且不超过 2000 字后重试。";
    case 401:
    case 403:
      return "当前整理服务未授权，请联系管理员检查服务配置。";
    case 404:
      return "当前会话已过期，请新建整理后重试。";
    case 409:
      if (error.detail?.includes("请求上限") || error.detail?.includes("回放窗口")) {
        return "当前会话无法继续安全重放，请新建整理后继续。";
      }
      return "会话刚刚被其他请求更新，请确认输入后重试。";
    case 426:
      return "当前入口未启用 HTTPS，请通过受保护的服务地址继续。";
    case 429:
      return "请求过于频繁，请稍后再试。";
    default:
      return error.status >= 500
        ? "整理服务暂时不可用，请稍后重试。"
        : `整理服务返回错误（HTTP ${error.status}），请稍后重试。`;
  }
}

function formatAuthError(error: unknown, mode: AuthMode) {
  if (!(error instanceof HttpResponseError)) {
    return "暂时无法连接账户服务，请检查网络后重试。";
  }
  if (error.status === 409 && mode === "register") {
    return "该用户名已被使用，请更换后重试。";
  }
  if (error.status === 401 && mode === "login") {
    return "用户名或密码错误，请重新输入。";
  }
  if (error.status === 400 || error.status === 422) {
    return "账户信息未通过校验，请检查用户名和密码格式。";
  }
  if (error.status === 429) {
    return "尝试次数过多，请稍后再试。";
  }
  if (error.status === 426) {
    return "当前入口未启用 HTTPS，请通过受保护的服务地址继续。";
  }
  return error.status >= 500
    ? "账户服务暂时不可用，请稍后重试。"
    : `账户服务返回错误（HTTP ${error.status}），请稍后重试。`;
}

function categoryLabel(category: string) {
  const labels: Record<string, string> = {
    disease: "疾病资料",
    drug: "药品资料",
    exam: "检查资料",
    department: "就诊方向",
    faq: "服务 FAQ",
  };
  return labels[category] || category;
}

function App() {
  const [authStatus, setAuthStatus] = useState<AuthStatus>("checking");
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [authBootstrapError, setAuthBootstrapError] = useState<string | null>(null);
  const [logoutPending, setLogoutPending] = useState(false);
  const [session, setSession] = useState<SessionState | null>(null);
  const [messages, setMessages] = useState<Message[]>([initialMessage]);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionStatus, setSessionStatus] = useState<SessionStatus>("idle");
  const [streaming, setStreaming] = useState(true);
  const [events, setEvents] = useState<PipelineEvent[]>([]);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [structuredResult, setStructuredResult] = useState<StructuredResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mobilePanel, setMobilePanel] = useState<"left" | "right" | null>(null);
  const [feedbackSent, setFeedbackSent] = useState<Record<string, "up" | "down">>({});
  const [feedbackPending, setFeedbackPending] = useState<Record<string, boolean>>({});
  const bottomRef = useRef<HTMLDivElement>(null);
  const generationRef = useRef(0);
  const activeSessionIdRef = useRef<string | null>(null);
  const sessionRequestRef = useRef<AbortController | null>(null);
  const chatRequestRef = useRef<AbortController | null>(null);
  const messageSequenceRef = useRef(0);
  const sessionCreateKeyRef = useRef<string | null>(null);
  const initialSessionStartedRef = useRef(false);
  const mountedRef = useRef(false);
  const pendingRetryRef = useRef<{ sessionId: string; text: string; requestId: string } | null>(null);
  const committedResponseIdsRef = useRef<Set<string>>(new Set());
  const leftPanelRef = useRef<HTMLElement>(null);
  const rightPanelRef = useRef<HTMLElement>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const closeTimerRef = useRef<number | null>(null);
  const authRequestRef = useRef<AbortController | null>(null);

  useEffect(() => {
    mountedRef.current = true;
    if (closeTimerRef.current !== null) {
      window.clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
    if (!initialSessionStartedRef.current) {
      initialSessionStartedRef.current = true;
      restoreAuthentication();
    }
    // React StrictMode replays this effect in development.  Defer aborting
    // until the next task so the replay can reuse the same in-flight request.
    return () => {
      mountedRef.current = false;
      if (closeTimerRef.current !== null) window.clearTimeout(closeTimerRef.current);
      closeTimerRef.current = window.setTimeout(() => {
        if (mountedRef.current) return;
        generationRef.current += 1;
        authRequestRef.current?.abort();
        sessionRequestRef.current?.abort();
        chatRequestRef.current?.abort();
      }, 0);
    };
  }, []);

  useEffect(() => {
    const mobileLayout = window.matchMedia("(max-width: 1020px)");
    const closeWhenDesktop = (event: MediaQueryListEvent) => {
      if (event.matches) return;
      restoreFocusRef.current = null;
      setMobilePanel(null);
    };
    mobileLayout.addEventListener("change", closeWhenDesktop);
    return () => mobileLayout.removeEventListener("change", closeWhenDesktop);
  }, []);

  useEffect(() => {
    if (!mobilePanel) return;
    const panel = mobilePanel === "left" ? leftPanelRef.current : rightPanelRef.current;
    if (!panel) return;
    const focusableSelector = [
      "button:not([disabled])",
      "a[href]",
      "input:not([disabled])",
      "textarea:not([disabled])",
      "select:not([disabled])",
      "[tabindex]:not([tabindex='-1'])",
    ].join(",");
    const focusable = () => Array.from(panel.querySelectorAll<HTMLElement>(focusableSelector))
      .filter((element) => element.getClientRects().length > 0);
    const focusFirst = () => {
      const target = focusable()[0] || panel;
      target.focus();
    };
    const focusTimer = window.setTimeout(focusFirst, 0);
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeMobilePanel();
        return;
      }
      if (event.key !== "Tab") return;
      const elements = focusable();
      if (!elements.length) {
        event.preventDefault();
        panel.focus();
        return;
      }
      const first = elements[0];
      const last = elements[elements.length - 1];
      if (!panel.contains(document.activeElement)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.clearTimeout(focusTimer);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [mobilePanel]);

  useEffect(() => {
    const background = [
      document.querySelector<HTMLElement>(".topbar"),
      document.querySelector<HTMLElement>(".chat-workspace"),
    ].filter((element): element is HTMLElement => Boolean(element));
    for (const element of background) {
      if (mobilePanel) {
        element.setAttribute("inert", "");
      } else {
        element.removeAttribute("inert");
      }
    }
    return () => {
      for (const element of background) {
        element.removeAttribute("inert");
      }
    };
  }, [mobilePanel]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  function resetWorkspaceState() {
    activeSessionIdRef.current = null;
    pendingRetryRef.current = null;
    committedResponseIdsRef.current.clear();
    sessionCreateKeyRef.current = null;
    setSession(null);
    setSessionStatus("idle");
    setMessages([{ ...initialMessage, timestamp: Date.now() }]);
    setDraft("");
    setLoading(false);
    setError(null);
    setEvents([]);
    setCitations([]);
    setStructuredResult(null);
    setFeedbackSent({});
    setFeedbackPending({});
    setMobilePanel(null);
  }

  function returnToAuthentication(message: string) {
    generationRef.current += 1;
    sessionRequestRef.current?.abort();
    chatRequestRef.current?.abort();
    sessionRequestRef.current = null;
    chatRequestRef.current = null;
    resetWorkspaceState();
    setAuthUser(null);
    setAuthStatus("anonymous");
    setAuthBootstrapError(message);
  }

  function enterWorkspace(user: AuthUser) {
    resetWorkspaceState();
    setAuthUser(user);
    setAuthStatus("authenticated");
    setAuthBootstrapError(null);
    const createKey = createRequestId("session");
    sessionCreateKeyRef.current = createKey;
    startSession(createKey);
  }

  function restoreAuthentication() {
    authRequestRef.current?.abort();
    const controller = new AbortController();
    authRequestRef.current = controller;
    setAuthStatus("checking");
    setAuthBootstrapError(null);
    void (async () => {
      try {
        const response = await apiFetch("/api/auth/me", { signal: controller.signal });
        if (response.status === 401) {
          if (authRequestRef.current !== controller || controller.signal.aborted) return;
          setAuthUser(null);
          setAuthStatus("anonymous");
          return;
        }
        await ensureResponseOk(response);
        const user = (await response.json()) as AuthUser;
        if (!user.username || authRequestRef.current !== controller || controller.signal.aborted) return;
        enterWorkspace(user);
      } catch (requestError) {
        if (isAbortError(requestError) || authRequestRef.current !== controller) return;
        setAuthUser(null);
        setAuthStatus("anonymous");
        setAuthBootstrapError(requestError instanceof HttpResponseError
          ? formatAuthError(requestError, "login")
          : "暂时无法连接账户服务，请检查网络后重试。");
      } finally {
        if (authRequestRef.current === controller) authRequestRef.current = null;
      }
    })();
  }

  async function logout() {
    if (logoutPending) return;
    setLogoutPending(true);
    setError(null);
    try {
      const response = await apiFetch("/api/auth/logout", { method: "POST" });
      if (response.status !== 401) await ensureResponseOk(response);
      returnToAuthentication("");
      setAuthBootstrapError(null);
    } catch (requestError) {
      setError(requestError instanceof HttpResponseError
        ? `退出失败：${formatAuthError(requestError, "login")}`
        : "退出失败，请检查网络后重试。");
    } finally {
      setLogoutPending(false);
    }
  }

  function requestIsCurrent(generation: number, controller: AbortController, sessionId?: string) {
    return generationRef.current === generation
      && !controller.signal.aborted
      && (!sessionId || activeSessionIdRef.current === sessionId);
  }

  function nextMessageId(prefix: string) {
    messageSequenceRef.current += 1;
    return `${prefix}-${Date.now()}-${messageSequenceRef.current}`;
  }

  function addMessage(message: Omit<Message, "timestamp"> & { timestamp?: number }) {
    setMessages((current) => [
      ...current,
      { ...message, timestamp: message.timestamp ?? Date.now() },
    ]);
  }

  function openMobilePanel(panel: "left" | "right", trigger?: HTMLElement) {
    if (mobilePanel === panel) {
      closeMobilePanel();
      return;
    }
    restoreFocusRef.current = trigger || (document.activeElement instanceof HTMLElement ? document.activeElement : null);
    setMobilePanel(panel);
  }

  function closeMobilePanel() {
    const restoreTarget = restoreFocusRef.current;
    restoreFocusRef.current = null;
    setMobilePanel(null);
    if (restoreTarget) {
      window.setTimeout(() => restoreTarget.focus(), 0);
    }
  }

  function handleComposerKeyDown(event: ReactKeyboardEvent<HTMLTextAreaElement>) {
    if (event.nativeEvent.isComposing || event.nativeEvent.keyCode === 229) return;
    if (event.key !== "Enter" || event.shiftKey) return;
    event.preventDefault();
    void sendMessage();
  }

  function startSession(createKey = sessionCreateKeyRef.current || createRequestId("session")) {
    sessionCreateKeyRef.current = createKey;
    const generation = ++generationRef.current;
    sessionRequestRef.current?.abort();
    chatRequestRef.current?.abort();
    chatRequestRef.current = null;
    pendingRetryRef.current = null;
    activeSessionIdRef.current = null;
    setSession(null);
    setSessionStatus("creating");
    setLoading(false);
    setError(null);

    const controller = new AbortController();
    sessionRequestRef.current = controller;
    void (async () => {
      try {
        const response = await apiFetch("/api/sessions", {
          method: "POST",
          headers: apiHeaders({
            // The key lets a StrictMode effect replay resolve to one server session.
            "Idempotency-Key": createKey,
          }),
          signal: controller.signal,
        });
        await ensureResponseOk(response);
        const body = (await response.json()) as { state?: SessionState };
        if (!body.state?.session_id || !requestIsCurrent(generation, controller)) return;
        activeSessionIdRef.current = body.state.session_id;
        setSession(body.state);
        setSessionStatus("ready");
      } catch (requestError) {
        if (isAbortError(requestError) || !requestIsCurrent(generation, controller)) return;
        if (requestError instanceof HttpResponseError && requestError.status === 401) {
          returnToAuthentication("登录状态已过期，请重新登录。");
          return;
        }
        activeSessionIdRef.current = null;
        setSession(null);
        setSessionStatus("error");
        setError(requestError instanceof HttpResponseError ? formatHttpError(requestError) : "整理服务暂不可用，请稍后重试。");
      } finally {
        if (sessionRequestRef.current === controller) sessionRequestRef.current = null;
      }
    })();
  }

  async function sendMessage(event?: FormEvent, prompt?: string) {
    event?.preventDefault();
    const text = (prompt ?? draft).trim();
    if (!text || loading || chatRequestRef.current) return;
    if (text.length > 2000) {
      setError("输入过长，请将描述控制在 2000 字以内后重试。");
      return;
    }
    const sessionId = activeSessionIdRef.current;
    if (!sessionId) {
      setError("整理会话仍在建立，请稍候再试。");
      return;
    }
    const generation = generationRef.current;
    const controller = new AbortController();
    chatRequestRef.current = controller;
    const userMessageId = nextMessageId("user");
    const draftBeforeSubmit = draft;
    const isDraftSubmit = prompt === undefined;
    const retry = pendingRetryRef.current;
    const requestId = retry?.sessionId === sessionId && retry.text === text
      ? retry.requestId
      : createRequestId("turn");
    pendingRetryRef.current = { sessionId, text, requestId };
    if (isDraftSubmit) setDraft("");
    setError(null);
    addMessage({ id: userMessageId, role: "user", content: text, sessionId, requestId });
    setLoading(true);
    setEvents([]);
    try {
      const response = await apiFetch(`/api/chat?stream=${streaming}`, {
        method: "POST",
        headers: apiHeaders({
          "Content-Type": "application/json",
          "Idempotency-Key": requestId,
        }),
        body: JSON.stringify({ session_id: sessionId, message: text, request_id: requestId }),
        signal: controller.signal,
      });
      await ensureResponseOk(response);
      if (streaming && response.body) {
        const body = await consumeStream(response.body, generation, sessionId, controller);
        applyFinal(body, generation, sessionId, controller);
      } else {
        const body = (await response.json()) as ChatResponseBody;
        applyFinal(body, generation, sessionId, controller);
      }
      pendingRetryRef.current = null;
    } catch (requestError) {
      if (isAbortError(requestError) || !requestIsCurrent(generation, controller, sessionId)) return;
      if (requestError instanceof HttpResponseError && requestError.status === 401) {
        returnToAuthentication("登录状态已过期，请重新登录。");
        return;
      }
      if (!(requestError instanceof HttpResponseError)) {
        const recovered = await recoverCommittedResult(sessionId, requestId, controller);
        if (recovered && requestIsCurrent(generation, controller, sessionId)) {
          applyFinal(recovered, generation, sessionId, controller);
          pendingRetryRef.current = null;
          setError(null);
          return;
        }
      }
      setMessages((current) => current.filter((message) => message.id !== userMessageId));
      if (isDraftSubmit) setDraft((current) => current || draftBeforeSubmit);
      // Keep the same request id for an explicit retry.  The API can then
      // return an already committed result instead of advancing the turn.
      pendingRetryRef.current = { sessionId, text, requestId };
      setError(requestError instanceof HttpResponseError
        ? formatHttpError(requestError)
        : "暂时无法连接整理服务，请检查网络或稍后重试。你的输入已保留。");
    } finally {
      if (requestIsCurrent(generation, controller, sessionId)) setLoading(false);
      if (chatRequestRef.current === controller) chatRequestRef.current = null;
    }
  }

  async function consumeStream(
    body: ReadableStream<Uint8Array>,
    generation: number,
    sessionId: string,
    controller: AbortController,
  ): Promise<ChatResponseBody> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalPayload: ChatResponseBody | null = null;
    while (true) {
      const { value, done } = await reader.read();
      if (!requestIsCurrent(generation, controller, sessionId)) {
        await reader.cancel();
        throw new DOMException("The chat request was superseded", "AbortError");
      }
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";
      if (done && buffer.trim()) {
        frames.push(buffer);
        buffer = "";
      }
      for (const frame of frames) {
        const eventLine = frame.split("\n").find((line) => line.startsWith("event:"));
        const dataLine = frame.split("\n").find((line) => line.startsWith("data:"));
        if (!eventLine || !dataLine) continue;
        const type = eventLine.replace("event:", "").trim();
        const payload = JSON.parse(dataLine.replace("data:", "").trim());
        if (type === "node") {
          setEvents((current) => [...current, payload]);
        } else if (type === "final") {
          finalPayload = payload as ChatResponseBody;
        } else if (type === "error") {
          const status = typeof payload?.status === "number" ? payload.status : 500;
          const detail = typeof payload?.message === "string" ? payload.message : undefined;
          throw new HttpResponseError(status, detail);
        }
      }
      // A complete final frame is the application-level commit marker.  Do
      // not wait for the transport to close: the socket may drop after the
      // payload was received, and discarding it would create an orphaned turn.
      if (finalPayload) return finalPayload;
      if (done) break;
    }
    if (!finalPayload && requestIsCurrent(generation, controller, sessionId)) {
      throw new Error("Stream ended before the final answer");
    }
    if (!finalPayload) throw new Error("Stream was cancelled before the final answer");
    return finalPayload;
  }

  async function recoverCommittedResult(
    sessionId: string,
    requestId: string,
    controller: AbortController,
  ): Promise<ChatResponseBody | null> {
    for (let attempt = 0; attempt < 3; attempt += 1) {
      if (controller.signal.aborted) return null;
      try {
        const response = await apiFetch(
          `/api/sessions/${encodeURIComponent(sessionId)}/requests/${encodeURIComponent(requestId)}`,
          { headers: apiHeaders(), signal: controller.signal },
        );
        if (response.ok) return (await response.json()) as ChatResponseBody;
        if (response.status !== 404) return null;
      } catch (recoveryError) {
        if (isAbortError(recoveryError)) return null;
      }
      if (attempt < 2) await new Promise((resolve) => window.setTimeout(resolve, 120));
    }
    return null;
  }

  function applyFinal(body: ChatResponseBody, generation: number, sessionId: string, controller: AbortController) {
    if (!requestIsCurrent(generation, controller, sessionId)) return;
    if (
      !body.state?.session_id
      || body.state.session_id !== sessionId
      || (body.session_id && body.session_id !== sessionId)
      || (body.session_id && body.state.session_id !== body.session_id)
    ) {
      throw new Error("Chat response has an invalid session identity");
    }
    const responseId = body.response_id || `${body.state.session_id}:turn:${body.state.turn_count}`;
    if (committedResponseIdsRef.current.has(responseId)) return;
    committedResponseIdsRef.current.add(responseId);
    activeSessionIdRef.current = body.state.session_id;
    setSession(body.state);
    setCitations(body.citations || []);
    setStructuredResult(body.structured_result || null);
    addMessage({
      id: responseId,
      role: "assistant",
      content: body.answer,
      citations: body.citations || [],
      structuredResult: body.structured_result,
      latency: body.latency_ms,
      sessionId: body.state.session_id,
    });
  }

  function newConversation() {
    setMessages([{ ...initialMessage, timestamp: Date.now() }]);
    setDraft("");
    setCitations([]);
    setStructuredResult(null);
    setEvents([]);
    setFeedbackSent({});
    setFeedbackPending({});
    committedResponseIdsRef.current.clear();
    restoreFocusRef.current = null;
    closeMobilePanel();
    sessionCreateKeyRef.current = createRequestId("session");
    startSession(sessionCreateKeyRef.current);
  }

  async function sendFeedback(messageId: string, rating: "up" | "down") {
    const message = messages.find((item) => item.id === messageId);
    const sessionId = message?.sessionId || activeSessionIdRef.current;
    if (!sessionId || feedbackSent[messageId] || feedbackPending[messageId]) return;
    const generation = generationRef.current;
    setFeedbackPending((current) => ({ ...current, [messageId]: true }));
    try {
      const response = await apiFetch("/api/feedback", {
        method: "POST",
        headers: apiHeaders({
          "Content-Type": "application/json",
          // Retried clicks for one answer/rating must remain one metric event.
          "Idempotency-Key": `${sessionId}:${messageId}:${rating}`,
        }),
        body: JSON.stringify({ session_id: sessionId, message_id: messageId, rating }),
      });
      await ensureResponseOk(response);
      if (generationRef.current === generation) {
        setFeedbackSent((current) => ({ ...current, [messageId]: rating }));
      }
    } catch (requestError) {
      if (generationRef.current === generation) {
        if (requestError instanceof HttpResponseError && requestError.status === 401) {
          returnToAuthentication("登录状态已过期，请重新登录。");
          return;
        }
        setError(requestError instanceof HttpResponseError
          ? `反馈提交失败：${formatHttpError(requestError)}`
          : "反馈提交失败，请稍后重试。");
      }
    } finally {
      setFeedbackPending((current) => {
        if (!current[messageId]) return current;
        const next = { ...current };
        delete next[messageId];
        return next;
      });
    }
  }

  if (authStatus !== "authenticated" || !authUser) {
    return (
      <AuthScreen
        status={authStatus}
        bootstrapError={authBootstrapError}
        onAuthenticated={enterWorkspace}
        onRetryConnection={restoreAuthentication}
      />
    );
  }

  const riskAssessed = Boolean(session && (session.risk_assessed ?? session.turn_count > 0));
  const risk: DisplayRiskLevel = riskAssessed && session?.risk_level ? session.risk_level : "unknown";
  const profile = session?.profile ?? {};
  const serviceStatusLabel = sessionStatus === "error"
    ? "服务不可用"
    : !session
      ? "建立安全会话"
      : session.network_enabled === false
        ? "知识服务受限"
        : "整理服务在线";
  const riskLabel = risk === "high"
    ? "需要立即就医"
    : risk === "watch"
      ? "持续观察"
      : risk === "low"
        ? "暂未见高风险"
        : "风险待评估";
  const riskHeading = risk === "high"
    ? "检测到高风险信号"
    : risk === "watch"
      ? "建议持续观察"
      : risk === "low"
        ? "当前风险较低"
        : "尚未完成安全筛查";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark"><HeartPulse size={18} strokeWidth={2.4} /></div>
          <div>
            <div className="brand-name">MedGuide</div>
            <div className="brand-subtitle">健康信息整理与就医指引</div>
          </div>
        </div>
        <div className="topbar-meta">
          <span className="service-status" role="status"><span className="status-dot" />{serviceStatusLabel}</span>
          <div className="account-cluster">
            <span className="account-identity"><UserRound size={15} /><span className="account-name">{authUser.username}</span></span>
            <button className="logout-button" onClick={() => void logout()} disabled={logoutPending} aria-label="退出登录" title="退出登录">
              <LogOut size={15} /><span>{logoutPending ? "退出中" : "退出"}</span>
            </button>
          </div>
          <button
            className="icon-button mobile-trigger"
            onClick={(event) => openMobilePanel("right", event.currentTarget)}
            aria-label="打开上下文面板"
            aria-controls="context-panel"
            aria-expanded={mobilePanel === "right"}
          >
            <PanelRight size={18} />
          </button>
        </div>
      </header>

      <div className="workspace">
        {mobilePanel && <button className="mobile-panel-backdrop" onClick={closeMobilePanel} aria-label="关闭面板" />}
        <aside
          id="session-panel"
          ref={leftPanelRef}
          className={`left-rail ${mobilePanel === "left" ? "mobile-visible" : ""}`}
          role={mobilePanel === "left" ? "dialog" : undefined}
          aria-modal={mobilePanel === "left" ? true : undefined}
          aria-labelledby="session-panel-title"
          tabIndex={mobilePanel === "left" ? -1 : undefined}
        >
          <div className="rail-heading">
            <span className="eyebrow" id="session-panel-title">整理管理</span>
            <div className="rail-heading-actions">
              <button className="icon-button" onClick={newConversation} aria-label="新建整理"><Plus size={17} /></button>
              <button className="icon-button close-mobile" onClick={closeMobilePanel} aria-label="关闭会话列表"><X size={16} /></button>
            </div>
          </div>
          <button className="new-session-button" onClick={newConversation}><Plus size={16} />新建整理</button>
          <div className="rail-section-label"><MessageCircle size={13} />当前整理</div>
          <div className="conversation-list">
            <button className="conversation-item active" onClick={closeMobilePanel}>
              <span className="conversation-icon"><MessageCircle size={16} /></span>
              <span className="conversation-copy"><strong>当前整理</strong><span>{messages.length > 1 ? `${messages.length - 1} 条消息` : "等待输入"}</span></span>
            </button>
            <div className="conversation-empty">暂无其他可恢复会话</div>
          </div>
          <div className="rail-footer">
            <div className="privacy-line"><ShieldCheck size={15} /><span>已启用安全会话<br />请勿输入姓名、证件号等身份信息</span></div>
          </div>
        </aside>

        <main className="chat-workspace">
          <div className="chat-header">
            <div>
              <div className="eyebrow">健康信息整理</div>
              <h1>信息整理</h1>
            </div>
            <div className="chat-header-actions">
              <span className={`risk-chip ${risk}`}><span className="risk-chip-dot" />{riskLabel}</span>
              <button
                className="secondary-button"
                onClick={(event) => openMobilePanel("left", event.currentTarget)}
                aria-controls="session-panel"
                aria-expanded={mobilePanel === "left"}
              ><LayoutDashboard size={15} />会话</button>
            </div>
          </div>

           <div className="patient-strip">
            <div className="patient-avatar"><UserRound size={18} /></div>
             <div className="patient-data"><span className="patient-name">{authUser.username}</span><span className="patient-id">会话 {session?.session_id ?? (sessionStatus === "error" ? "不可用" : "准备中")}</span></div>
            <div className="patient-facts">
              <span><span className="fact-label">年龄</span>{profile.age ? `${profile.age} 岁` : "待补充"}</span>
              <span><span className="fact-label">性别</span>{profile.sex || "待补充"}</span>
              <span><span className="fact-label">症状</span>{profile.chief_complaint || "待采集"}</span>
            </div>
            <div className="turn-count"><span>整理轮次</span><strong>{session?.turn_count ?? 0}</strong></div>
          </div>

          <div className="messages-scroller">
            <div className="message-date"><span />{formatDateDivider(messages[0]?.timestamp ?? Date.now())}<span /></div>
            {messages.map((message) => (
              <article className={`message-row ${message.role}`} key={message.id}>
                {message.role === "assistant" && <div className="message-avatar assistant-avatar"><Sparkles size={16} /></div>}
                <div className="message-content">
                  <div className="message-meta"><strong>{message.role === "assistant" ? "MedGuide" : "你"}</strong><span>{message.role === "assistant" ? "健康信息助手" : authUser.username}</span><time dateTime={new Date(message.timestamp).toISOString()}>{formatTime(new Date(message.timestamp))}</time></div>
                  <div className="message-bubble">{message.content.split("\n").map((line, index) => <p key={`${message.id}-${index}`}>{line || "\u00a0"}</p>)}</div>
                  {message.role === "assistant" && message.citations && message.citations.length > 0 && (
                    <div className="inline-citations"><BookOpen size={13} /><span>已引用 {message.citations.length} 条知识库资料</span><ArrowUpRight size={12} /></div>
                  )}
                  {message.role === "assistant" && message.latency !== undefined && <div className="message-latency"><Clock3 size={12} />响应 {Math.round(message.latency)} ms</div>}
                  {message.role === "assistant" && message.id !== "welcome" && <div className="message-feedback"><span>这条回答有帮助吗？</span><button className={feedbackSent[message.id] === "up" ? "selected" : ""} onClick={() => void sendFeedback(message.id, "up")} disabled={feedbackPending[message.id]} aria-label="回答有帮助" title="回答有帮助"><ThumbsUp size={13} /></button><button className={feedbackSent[message.id] === "down" ? "selected" : ""} onClick={() => void sendFeedback(message.id, "down")} disabled={feedbackPending[message.id]} aria-label="回答需要改进" title="回答需要改进"><ThumbsDown size={13} /></button></div>}
                </div>
              </article>
            ))}
            {loading && (
              <article className="message-row assistant loading-row">
                <div className="message-avatar assistant-avatar"><Sparkles size={16} /></div>
                <div className="message-content"><div className="message-meta"><strong>MedGuide</strong><span>正在编排整理流程</span></div><div className="loading-bubble"><span /><span /><span /></div></div>
              </article>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="composer-wrap">
             <form className="composer" onSubmit={(event) => void sendMessage(event)}>
              <textarea value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={handleComposerKeyDown} placeholder="描述你的症状，或询问用药须知、报告解读和就诊方向…" rows={2} maxLength={2000} disabled={loading} />
              <div className="composer-bottom"><span className="composer-hint"><Info size={13} />请勿输入身份信息，内容将用于当前整理</span><button className="send-button" type="submit" disabled={!draft.trim() || loading || !session} aria-label="发送消息"><Send size={17} /></button></div>
            </form>
            <div className="composer-disclaimer">MedGuide 提供健康信息整理与就医指引，不替代医生诊断、处方或急救服务。</div>
          </div>
        </main>

        <aside
          id="context-panel"
          ref={rightPanelRef}
          className={`right-panel ${mobilePanel === "right" ? "mobile-visible" : ""}`}
          role={mobilePanel === "right" ? "dialog" : undefined}
          aria-modal={mobilePanel === "right" ? true : undefined}
          aria-labelledby="context-panel-title"
          tabIndex={mobilePanel === "right" ? -1 : undefined}
        >
          <div className="panel-topline"><span className="eyebrow" id="context-panel-title">可追溯上下文</span><button className="icon-button close-mobile" onClick={closeMobilePanel} aria-label="关闭面板"><X size={16} /></button></div>
          <section className={`risk-panel ${risk}`}>
            <div className="risk-heading"><div className="risk-icon">{risk === "high" ? <AlertTriangle size={17} /> : risk === "unknown" ? <Info size={17} /> : <ShieldCheck size={17} />}</div><div><span className="section-kicker">安全筛查</span><h2>{riskHeading}</h2></div></div>
            {session?.risk_flags?.length
              ? <div className="risk-flags">{session.risk_flags.map((flag) => <span key={flag}>{flag}</span>)}</div>
              : <p>{riskAssessed ? "已完成规则筛查，继续补充信息后会动态更新。" : "尚未收到有效信息，当前不能给出风险结论。"}</p>}
            <div className="risk-meter"><span className="meter-label">风险等级</span><div className="meter-track"><span className="meter-fill" /></div><strong>{risk === "high" ? "HIGH" : risk === "watch" ? "WATCH" : risk === "low" ? "LOW" : "UNKNOWN"}</strong></div>
          </section>

          <section className="panel-section profile-section"><div className="section-heading"><div><span className="section-kicker">结构化状态</span><h2>整理要素</h2></div><Activity size={16} /></div><div className="profile-grid"><ProfileField label="主要症状" value={profile.chief_complaint as string} /><ProfileField label="持续时间" value={profile.duration as string} /><ProfileField label="伴随症状" value={Array.isArray(profile.associated_symptoms) ? profile.associated_symptoms.join("、") : undefined} /><ProfileField label="既往史" value={profile.history as string} /></div>{session?.summary && <div className="summary-box"><span>整理摘要</span><p>{session.summary}</p></div>}</section>

          <section className="panel-section pipeline-section"><div className="section-heading"><div><span className="section-kicker">可观察工作流</span><h2>节点进度</h2></div><span className="pipeline-count">{events.filter((event) => event.node).length}/9</span></div><div className="pipeline-list">{["清理输入", "抽取信息要素", "识别咨询意图", "筛查风险信号", "检索可信知识", "执行结构化查询", "生成可读建议", "执行安全审查", "整理引用与摘要"].map((label, index) => { const complete = events.length > index; return <div className={`pipeline-item ${complete ? "complete" : ""}`} key={label}><span className="pipeline-index">{complete ? <Check size={11} /> : String(index + 1).padStart(2, "0")}</span><span>{label}</span>{complete && <small>{events[index]?.elapsed_ms ?? 0}ms</small>}</div>; })}</div></section>

          <section className="panel-section evidence-section"><div className="section-heading"><div><span className="section-kicker">知识库命中</span><h2>参考资料 <span>{citations.length}</span></h2></div><BookOpen size={16} /></div>{citations.length ? <div className="evidence-list">{citations.map((citation) => { const sourceUrl = safeCitationSourceUrl(citation.source_url); return <div className="evidence-item" key={citation.id}><div className="evidence-item-top"><span className="evidence-type">{categoryLabel(citation.category)}</span><span className="evidence-score">{Math.round(citation.score * 100)}%</span></div><strong>{citation.title}</strong><p>{citation.snippet}</p><div className="evidence-source"><span>{citation.source} · {citation.updated_at}</span>{sourceUrl && <a href={sourceUrl} target="_blank" rel="noreferrer" aria-label={`查看来源：${citation.title}`}>查看原文<ArrowUpRight size={11} /></a>}</div></div>; })}</div> : <div className="empty-state"><BookOpen size={18} /><p>发送一条症状或知识问题，相关引用会出现在这里。</p></div>}</section>

          {structuredResult && <section className="panel-section query-section"><div className="section-heading"><div><span className="section-kicker">只读数据查询</span><h2>结构化结果</h2></div><ShieldCheck size={16} /></div>{structuredResult.blocked ? <p className="query-blocked">{structuredResult.reason}</p> : <div className="query-table">{structuredResult.rows.map((row, index) => <div className="query-row" key={index}>{Object.entries(row).map(([key, value]) => <span key={key}><small>{key}</small>{value}</span>)}</div>)}</div>}</section>}

          <div className="panel-footnote"><ShieldCheck size={14} /><span>回答经过规则安全审查<br />来源版本和查询范围可追溯</span></div>
        </aside>
      </div>
      {error && <div className="toast" role="status"><Info size={15} />{error}<button onClick={() => setError(null)} aria-label="关闭提示"><X size={14} /></button></div>}
    </div>
  );
}

type AuthScreenProps = {
  status: AuthStatus;
  bootstrapError: string | null;
  onAuthenticated: (user: AuthUser) => void;
  onRetryConnection: () => void;
};

function AuthScreen({ status, bootstrapError, onAuthenticated, onRetryConnection }: AuthScreenProps) {
  const [mode, setMode] = useState<AuthMode>("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [passwordConfirmation, setPasswordConfirmation] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [pending, setPending] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const requestRef = useRef<AbortController | null>(null);

  useEffect(() => () => requestRef.current?.abort(), []);

  function selectMode(nextMode: AuthMode) {
    if (pending || nextMode === mode) return;
    setMode(nextMode);
    setPassword("");
    setPasswordConfirmation("");
    setShowPassword(false);
    setFormError(null);
  }

  async function submitAuthentication(event: FormEvent) {
    event.preventDefault();
    if (pending) return;

    const validation = validateAuthCredentials(
      username,
      password,
      mode === "register" ? passwordConfirmation : undefined,
    );
    if (validation.error) {
      setFormError(validation.error);
      return;
    }

    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setPending(true);
    setFormError(null);
    try {
      const response = await apiFetch(`/api/auth/${mode}`, {
        method: "POST",
        headers: apiHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ username: validation.username, password }),
        signal: controller.signal,
      });
      await ensureResponseOk(response);
      const user = (await response.json()) as AuthUser;
      if (!user.username || controller.signal.aborted) return;
      onAuthenticated(user);
    } catch (requestError) {
      if (isAbortError(requestError) || controller.signal.aborted) return;
      setFormError(formatAuthError(requestError, mode));
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setPending(false);
      }
    }
  }

  const visibleError = formError || bootstrapError;
  const serviceUnavailable = Boolean(!formError && (bootstrapError?.includes("连接账户服务") || bootstrapError?.includes("服务暂时不可用")));

  return (
    <main className="auth-shell">
      <section className="auth-context" aria-labelledby="auth-product-name">
        <div className="auth-brand-lockup">
          <div className="auth-brand-mark"><HeartPulse size={24} strokeWidth={2.3} /></div>
          <div><span>MEDGUIDE</span><small>健康信息整理与就医指引</small></div>
        </div>
        <div className="auth-product-copy">
           <span className="section-kicker">HEALTH INFORMATION ROUTING</span>
          <h1 id="auth-product-name">MedGuide</h1>
          <p>在一个受保护的工作区内整理症状、完成风险筛查并查看可追溯的健康资料。</p>
        </div>
        <div className="auth-safety-note"><ShieldCheck size={17} /><span>请勿使用姓名、证件号或联系方式作为用户名，也不要在整理过程中提交可识别个人身份的信息。</span></div>
        <p className="auth-medical-note">MedGuide 提供健康信息整理与就医指引，不替代医生诊断、处方或急救服务。</p>
      </section>

      <section className="auth-panel" aria-labelledby="auth-heading">
        {status === "checking" ? (
          <div className="auth-loading" role="status" aria-live="polite">
            <span className="auth-spinner" aria-hidden="true" />
            <div><h2 id="auth-heading">正在恢复安全会话</h2><p>请稍候，正在验证登录状态。</p></div>
          </div>
        ) : (
          <>
            <div className="auth-panel-heading">
              <span className="section-kicker">账户访问</span>
              <h2 id="auth-heading">{mode === "login" ? "登录 MedGuide" : "创建账户"}</h2>
              <p>{mode === "login" ? "使用你的 MedGuide 账户继续整理。" : "用户名将作为你的唯一账户标识。"}</p>
            </div>

            <div className="auth-mode-switch" aria-label="选择账户操作">
              <button type="button" aria-pressed={mode === "login"} className={mode === "login" ? "active" : ""} onClick={() => selectMode("login")} disabled={pending}><LogIn size={15} />登录</button>
              <button type="button" aria-pressed={mode === "register"} className={mode === "register" ? "active" : ""} onClick={() => selectMode("register")} disabled={pending}><UserPlus size={15} />注册</button>
            </div>

            <form className="auth-form" onSubmit={(event) => void submitAuthentication(event)} noValidate aria-busy={pending}>
              <label className="auth-field">
                <span>用户名</span>
                <input
                  name="username"
                  value={username}
                  onChange={(event) => { setUsername(event.target.value); setFormError(null); }}
                  autoComplete="username"
                  autoCapitalize="none"
                  spellCheck={false}
                  minLength={3}
                  maxLength={USERNAME_INPUT_MAX_LENGTH}
                  aria-describedby="username-hint"
                  autoFocus
                  disabled={pending}
                />
                <small id="username-hint">3–64 个字符，可使用中英文、数字及 . _ -</small>
              </label>

              <label className="auth-field">
                <span>密码</span>
                <span className="password-control">
                  <input
                    name="password"
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={(event) => { setPassword(event.target.value); setFormError(null); }}
                    autoComplete={mode === "login" ? "current-password" : "new-password"}
                    minLength={8}
                    maxLength={PASSWORD_INPUT_MAX_LENGTH}
                    aria-describedby="password-hint"
                    disabled={pending}
                  />
                  <button type="button" onClick={() => setShowPassword((current) => !current)} aria-label={showPassword ? "隐藏密码" : "显示密码"} title={showPassword ? "隐藏密码" : "显示密码"} disabled={pending}>{showPassword ? <EyeOff size={17} /> : <Eye size={17} />}</button>
                </span>
                <small id="password-hint">8–128 个字符</small>
              </label>

              {mode === "register" && (
                <label className="auth-field">
                  <span>确认密码</span>
                  <span className="password-control">
                    <input
                      name="passwordConfirmation"
                      type={showPassword ? "text" : "password"}
                      value={passwordConfirmation}
                      onChange={(event) => { setPasswordConfirmation(event.target.value); setFormError(null); }}
                      autoComplete="new-password"
                      minLength={8}
                      maxLength={PASSWORD_INPUT_MAX_LENGTH}
                      disabled={pending}
                    />
                    <LockKeyhole size={16} aria-hidden="true" />
                  </span>
                </label>
              )}

              {visibleError && (
                <div className="auth-error" role="alert">
                  <AlertTriangle size={16} />
                  <span>{visibleError}</span>
                  {serviceUnavailable && <button type="button" onClick={onRetryConnection}>重新检测</button>}
                </div>
              )}

              <button className="auth-submit" type="submit" disabled={pending || !username || !password || (mode === "register" && !passwordConfirmation)}>
                {pending ? <span className="button-spinner" aria-hidden="true" /> : mode === "login" ? <LogIn size={17} /> : <UserPlus size={17} />}
                {pending ? (mode === "login" ? "正在登录" : "正在创建") : mode === "login" ? "登录" : "创建并登录"}
              </button>
            </form>
          </>
        )}
      </section>
    </main>
  );
}

function ProfileField({ label, value }: { label: string; value?: string }) {
  return <div className="profile-field"><span>{label}</span><strong className={value ? "filled" : "pending"}>{value || "待补充"}</strong></div>;
}

export default App;
