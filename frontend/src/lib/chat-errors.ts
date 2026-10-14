import { ApiError } from '@/lib/http'

export type ChatErrorKind =
  | 'network'
  | 'auth'
  | 'quota'
  | 'rate_limit'
  | 'input'
  | 'grounding'
  | 'retrieval'
  | 'generic'

export type ClassifiedChatError = {
  kind: ChatErrorKind
  title: string
  message: string
  showLoginLink: boolean
}

function messageIncludesAny(text: string, needles: string[]): boolean {
  const lower = text.toLowerCase()
  return needles.some((needle) => lower.includes(needle.toLowerCase()))
}

// The AI SDK surfaces non-2xx stream responses as `new Error(<response body text>)`,
// so FastAPI's `{"detail": "..."}` payload arrives as a JSON string. Unwrap it.
function unwrapDetail(text: string): string {
  const trimmed = text.trim()
  if (!trimmed.startsWith('{')) {
    return text
  }

  try {
    const parsed: unknown = JSON.parse(trimmed)
    if (
      parsed !== null &&
      typeof parsed === 'object' &&
      'detail' in parsed &&
      typeof parsed.detail === 'string'
    ) {
      return parsed.detail
    }
  } catch {
    // Not JSON; fall through to the raw text.
  }

  return text
}

export function classifyChatError(error: Error): ClassifiedChatError {
  if (error instanceof ApiError) {
    if (error.isNetworkError) {
      return {
        kind: 'network',
        title: 'Connection problem',
        message:
          "Can't reach the server. Check your connection and that the backend is running.",
        showLoginLink: false,
      }
    }

    if (error.status === 401) {
      return {
        kind: 'auth',
        title: 'Session expired',
        message: 'Your sign-in session has expired. Please sign in again.',
        showLoginLink: true,
      }
    }

    if (error.status === 403) {
      return {
        kind: 'auth',
        title: 'Access denied',
        message: 'You do not have access to this conversation.',
        showLoginLink: false,
      }
    }
  }

  const text = unwrapDetail(error.message || '')

  if (messageIncludesAny(text, ['too many requests', 'rate limit'])) {
    return {
      kind: 'rate_limit',
      title: 'Slow down a moment',
      message: 'You are sending questions faster than the assistant allows. Wait a few seconds and try again.',
      showLoginLink: false,
    }
  }

  if (messageIncludesAny(text, ['message is too long'])) {
    return {
      kind: 'input',
      title: 'Question too long',
      message: text,
      showLoginLink: false,
    }
  }

  if (
    messageIncludesAny(text, [
      'insufficient_quota',
      'credit_balance_exhausted',
      'no credits remaining',
    ])
  ) {
    return {
      kind: 'quota',
      title: 'AI service quota exhausted',
      message:
        'The AI provider has no quota remaining right now. Try again shortly, or ask an administrator to check the AI gateway.',
      showLoginLink: false,
    }
  }

  if (
    messageIncludesAny(text, [
      'grounding',
      'citation',
      'verified against source',
      'source passages',
      'fully verify',
    ])
  ) {
    return {
      kind: 'grounding',
      title: 'Answer not verified',
      message:
        text ||
        'The answer could not be verified against source documents. Try rephrasing your question.',
      showLoginLink: false,
    }
  }

  if (messageIncludesAny(text, ['assistant run failed', 'retrieval', 'search'])) {
    return {
      kind: 'retrieval',
      title: 'Search failed',
      message: 'Search or analysis failed. Please try again.',
      showLoginLink: false,
    }
  }

  if (messageIncludesAny(text, ['failed to fetch', 'network', 'cors'])) {
    return {
      kind: 'network',
      title: 'Connection problem',
      message:
        "Can't reach the server. Check your connection and that the backend is running.",
      showLoginLink: false,
    }
  }

  return {
    kind: 'generic',
    title: 'Something went wrong',
    message: text || 'Something went wrong while sending your message.',
    showLoginLink: false,
  }
}
