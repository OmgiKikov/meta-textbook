import type React from "react"
import "./globals.css"
import { Inter } from "next/font/google"
import { ThemeProvider } from "@/components/ui/theme-provider"
import Script from "next/script"

const inter = Inter({ subsets: ["latin", "cyrillic"] })

export const metadata = {
  title: "Knowledge Graph",
  description: "Interactive knowledge graph visualization",
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="ru" suppressHydrationWarning>
      <head>
        <Script id="disable-devtools" strategy="afterInteractive">
          {`
            // Скрыть иконку React DevTools
            if (typeof window !== 'undefined') {
              const devtools = document.createElement('style');
              devtools.innerHTML = '.__react-devtools-hook { display: none !important; } .__react-devtools-component-selector { display: none !important; }';
              document.head.appendChild(devtools);
            }
          `}
        </Script>
      </head>
      <body className={inter.className} suppressHydrationWarning>
        <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}

import './globals.css'
