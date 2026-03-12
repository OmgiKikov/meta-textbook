"use client"

import { useState, useEffect } from "react"
import { X, Key, Save, Check } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"

interface ApiKeyModalProps {
  isOpen: boolean
  onClose: () => void
  className?: string
}

export function ApiKeyModal({ isOpen, onClose, className }: ApiKeyModalProps) {
  const [apiKey, setApiKey] = useState("")
  const [isSaved, setIsSaved] = useState(false)
  const [isVisible, setIsVisible] = useState(false)

  // Загружаем ключ из localStorage при открытии
  useEffect(() => {
    if (isOpen) {
      const savedKey = localStorage.getItem("openai_api_key") || ""
      setApiKey(savedKey)
      setIsVisible(true)
    } else {
      setIsVisible(false)
    }
  }, [isOpen])

  // Сохраняем API ключ
  const handleSaveApiKey = () => {
    localStorage.setItem("openai_api_key", apiKey.trim())
    setIsSaved(true)

    // Создаем пользовательское событие для уведомления других компонентов
    const event = new Event("apiKeyChanged")
    window.dispatchEvent(event)

    // Показываем уведомление об успешном сохранении на 2 секунды
    setTimeout(() => {
      setIsSaved(false)
    }, 2000)
  }

  // Очищаем API ключ
  const handleClearApiKey = () => {
    setApiKey("")
    localStorage.removeItem("openai_api_key")

    // Создаем пользовательское событие для уведомления других компонентов
    const event = new Event("apiKeyChanged")
    window.dispatchEvent(event)
  }

  if (!isOpen) return null

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/20 backdrop-blur-sm z-50 transition-opacity duration-300"
        style={{ opacity: isVisible ? 1 : 0 }}
        onClick={onClose}
      />

      {/* Modal */}
      <div
        className={cn(
          "fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[450px] bg-background border border-border rounded-lg shadow-lg z-50 transform transition-all duration-300",
          isVisible ? "opacity-100 scale-100" : "opacity-0 scale-95",
          className,
        )}
      >
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div className="flex items-center gap-2">
            <Key className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-medium">Настройки API</h2>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="p-6 space-y-4">
          <div className="space-y-2">
            <Label htmlFor="api-key" className="flex items-center gap-1">
              OpenAI API ключ
              <span className="text-xs text-destructive">*обязательно</span>
            </Label>
            <Input
              id="api-key"
              type="password"
              placeholder="sk-..."
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="font-mono"
            />
            <p className="text-xs text-muted-foreground">
              Ваш API ключ будет сохранен только в локальном хранилище вашего браузера и не будет отправлен на сервер.
            </p>
          </div>

          <Separator />
          
          <div className="bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 rounded-md p-3">
            <h3 className="text-sm font-medium text-amber-800 dark:text-amber-300 mb-1">Важно!</h3>
            <p className="text-sm text-amber-700 dark:text-amber-400">
              Без API ключа OpenAI функции чата и создания mind map не будут работать.
              Ключ используется для генерации ответов и mind map на основе вашего текста.
            </p>
          </div>

          <div className="space-y-2">
            <h3 className="text-sm font-medium">Информация</h3>
            <p className="text-sm text-muted-foreground">
              Для использования API OpenAI вам необходим действующий API ключ. Вы можете получить его в{" "}
              <a
                href="https://platform.openai.com/api-keys"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                личном кабинете OpenAI
              </a>
              .
            </p>
          </div>

          <div className="space-y-1">
            <h3 className="text-sm font-medium">Как получить API ключ:</h3>
            <ol className="text-sm text-muted-foreground space-y-1 list-decimal list-inside pl-1">
              <li>Создайте аккаунт на <a href="https://openai.com" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">OpenAI.com</a></li>
              <li>Перейдите в <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">раздел API Keys</a></li>
              <li>Нажмите "Create new secret key"</li>
              <li>Скопируйте сгенерированный ключ и вставьте в поле выше</li>
            </ol>
          </div>
        </div>

        <div className="flex items-center justify-between p-4 border-t border-border bg-muted/50">
          <Button variant="outline" onClick={handleClearApiKey}>
            Очистить
          </Button>
          <div className="flex items-center gap-2">
            {isSaved && (
              <span className="flex items-center text-sm text-green-600 dark:text-green-500">
                <Check className="h-4 w-4 mr-1" />
                Сохранено
              </span>
            )}
            <Button onClick={handleSaveApiKey} disabled={!apiKey.trim()}>
              <Save className="h-4 w-4 mr-2" />
              Сохранить
            </Button>
          </div>
        </div>
      </div>
    </>
  )
}
