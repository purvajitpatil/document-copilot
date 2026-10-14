import { lazy, Suspense } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'

import { ProtectedRoute } from '@/components/ProtectedRoute'
import { PublicRoute } from '@/components/PublicRoute'
import { Login } from '@/pages/Login'
import { SignUp } from '@/pages/SignUp'

// The chat UI pulls in markdown + syntax highlighting (large). Load it only after sign-in.
const ChatLayout = lazy(() =>
  import('@/components/chat/ChatLayout').then((m) => ({ default: m.ChatLayout })),
)
const ChatEmptyPage = lazy(() =>
  import('@/pages/chat/ChatEmptyPage').then((m) => ({ default: m.ChatEmptyPage })),
)
const ChatThreadPage = lazy(() =>
  import('@/pages/chat/ChatThreadPage').then((m) => ({ default: m.ChatThreadPage })),
)

function PageFallback() {
  return (
    <div className="flex min-h-svh items-center justify-center text-sm text-muted-foreground">
      Loading…
    </div>
  )
}

function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<PageFallback />}>
      <Routes>
        <Route
          path="/login"
          element={
            <PublicRoute>
              <Login />
            </PublicRoute>
          }
        />
        <Route
          path="/signup"
          element={
            <PublicRoute>
              <SignUp />
            </PublicRoute>
          }
        />
        <Route path="/" element={<Navigate to="/chats" replace />} />
        <Route
          path="/chats"
          element={
            <ProtectedRoute>
              <ChatLayout />
            </ProtectedRoute>
          }
        >
          <Route index element={<ChatEmptyPage />} />
          <Route path=":threadId" element={<ChatThreadPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/chats" replace />} />
      </Routes>
      </Suspense>
    </BrowserRouter>
  )
}

export default App
