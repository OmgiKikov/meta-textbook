"use client"

import { cn } from "@/lib/utils"

interface NodeBadgeProps {
  type: string
  className?: string
}

export function NodeBadge({ type, className }: NodeBadgeProps) {
  // Get icon and color based on node type
  const getIconAndColor = () => {
    switch (type.toLowerCase()) {
      case "белок":
      case "protein":
        return {
          icon: (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path
                d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"
                fill="currentColor"
              />
            </svg>
          ),
          color: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400",
        }
      case "днк":
      case "dna":
        return {
          icon: (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path
                d="M4 12c0-4.41 3.59-8 8-8s8 3.59 8 8-3.59 8-8 8-8-3.59-8-8zm2 0c0 3.31 2.69 6 6 6s6-2.69 6-6-2.69-6-6-6-6 2.69-6 6z"
                fill="currentColor"
              />
              <path d="M12 10l-2 2 2 2 2-2-2-2z" fill="currentColor" />
            </svg>
          ),
          color: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
        }
      case "атом":
      case "atom":
        return {
          icon: (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="12" cy="12" r="3" fill="currentColor" />
              <path
                d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8z"
                fill="currentColor"
                fillOpacity="0.5"
              />
            </svg>
          ),
          color: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400",
        }
      case "клетка":
      case "cell":
        return {
          icon: (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path
                d="M20 2H4c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 18H4V4h16v16zM6 6h5v5H6V6zm7 0h5v5h-5V6zM6 13h5v5H6v-5zm7 0h5v5h-5v-5z"
                fill="currentColor"
              />
            </svg>
          ),
          color: "bg-cyan-100 text-cyan-700 dark:bg-cyan-900/30 dark:text-cyan-400",
        }
      default:
        return {
          icon: (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path
                d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          ),
          color: "bg-gray-100 text-gray-700 dark:bg-gray-800/50 dark:text-gray-400",
        }
    }
  }

  const { icon, color } = getIconAndColor()

  return (
    <div className={cn("inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium", color, className)}>
      <span className="mr-1">{icon}</span>
      {type}
    </div>
  )
}
