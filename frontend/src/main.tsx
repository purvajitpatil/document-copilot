import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@/lib/env'
import './index.css'
import App from './App.tsx'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import { TooltipProvider } from '@/components/ui/tooltip'
import { Toaster } from '@/components/ui/sonner'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <TooltipProvider>
        <App />
        <Toaster position="top-center" richColors />
      </TooltipProvider>
    </ErrorBoundary>
  </StrictMode>,
)
