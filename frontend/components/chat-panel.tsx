"use client"

import React, { useState, useEffect, useRef, forwardRef, type KeyboardEvent, memo, useMemo } from "react"
import { X, Maximize2, Minimize2, MapIcon, ThumbsUp, ThumbsDown, Send, StopCircle, ChevronDown, Settings, AlertCircle, RefreshCw, Key, Trash2 } from 'lucide-react'
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { cn } from "@/lib/utils"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { useChat, type Message } from "ai/react"
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { useReducer } from "react"
import { chatReducer, initialState, extractNodeContext, generateExpansionSuggestions, isNewTopic } from './chat-utils'

interface ChatMessage {
  id: string
  role: "user" | "assistant" | "system"
  content: string
}

interface ChatPanelProps {
  isOpen: boolean
  onClose: () => void
  onMindMapGenerated: (data: { mindMapMarkdown: string }) => void
  onMindMapError?: (errorMessage: string, failedRequestParams: { latestMessage: Message; currentMindMapMarkdown: string | null }) => void
  onSwitchToMindMap: () => void
  className?: string
  onTypingChange?: (isTyping: boolean) => void
}

// Мемоизированный компонент сообщения чата
const ChatMessageItem = memo(({ message }: { message: Message }) => {
  // Удаляем JSON mind map из отображаемого сообщения и форматируем текст
  const { displayContent, formattedContent } = useMemo(() => {
    let content = message.content;
    const jsonMatch = content.match(/\`\`\`json\s*([\s\S]*?)\s*\`\`\`/);
    
    if (jsonMatch) {
      content = content.replace(jsonMatch[0], "");
    }
    
    // Простое форматирование для ассистента
    let formatted = content;
    if (message.role === "assistant") {
      // Предварительная обработка текста для решения проблемы с запятыми
      // Объединяем строки, начинающиеся с запятой или другого знака пунктуации с предыдущей строкой
      formatted = formatted.replace(/\n\s*([,\.;:])/g, ' $1');
      
      // Преобразуем жирный текст в заголовки
      formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      
      // Создаем параграфы из двойных переносов строк
      const paragraphs = formatted.split(/\n\n+/g);
      formatted = paragraphs.map(p => {
        // Заменяем одиночные переносы строк на пробелы
        const cleanedParagraph = p.replace(/\n/g, ' ');
        
        // Оборачиваем каждый параграф в <p> теги
        return `<p>${cleanedParagraph}</p>`;
      }).join('');
    }
    
    return { displayContent: content, formattedContent: formatted };
  }, [message.content, message.role]);
  
  return (
    <div
      className={cn(
        "flex",
        message.role === "user" ? "justify-end" : "justify-start",
      )}
    >
      <div
        className={cn("flex max-w-[80%]", message.role === "user" ? "flex-row-reverse" : "flex-row")}
      >
        <Avatar className="h-8 w-8 mt-1">
          {message.role === "user" ? (
            <AvatarFallback className="bg-primary/10 text-primary">У</AvatarFallback>
          ) : (
            <AvatarFallback className="bg-secondary/10 text-secondary">ИИ</AvatarFallback>
          )}
        </Avatar>
        <div className={cn("mx-2", message.role === "user" ? "text-right" : "text-left")}>
          <div
            className={cn(
              "px-3 py-2 rounded-lg",
              message.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted",
            )}
          >
            {message.role === "assistant" ? (
              <div 
                className="text-sm message-content" 
                dangerouslySetInnerHTML={{ __html: formattedContent }}
              />
            ) : (
              <p className="text-sm whitespace-pre-line">{displayContent}</p>
            )}
          </div>
          <div className="flex items-center mt-1">
            <p className="text-xs text-muted-foreground">{new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</p>
            {message.role === "assistant" && (
              <div className="flex ml-2">
                <Button variant="ghost" size="icon" className="h-5 w-5 hover:bg-primary/10">
                  <ThumbsUp className="h-3 w-3" />
                </Button>
                <Button variant="ghost" size="icon" className="h-5 w-5 hover:bg-destructive/10">
                  <ThumbsDown className="h-3 w-3" />
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}, (prevProps, nextProps) => {
  // Оптимизированная функция сравнения для memo
  // Перерендерим только если содержимое сообщения изменилось
  return prevProps.message.content === nextProps.message.content && 
         prevProps.message.id === nextProps.message.id;
});

ChatMessageItem.displayName = "ChatMessageItem";

/**
 * Встраивает новую ветку (branchMarkdown) в существующий mindmap (currentMarkdown) по nodeName.
 * Если такой узел уже есть, заменяет его подузлы на новые, иначе добавляет новую ветку.
 */
function mergeBranch(currentMarkdown: string, nodeName: string, branchMarkdown: string): string {
  if (!currentMarkdown || !nodeName || !branchMarkdown) return currentMarkdown || branchMarkdown;

  const lines = currentMarkdown.split('\n');
  const branchLines = branchMarkdown.split('\n');

  // Найти строку с нужным nodeName и определить уровень (###, #### и т.д.)
  const nodeLineIndex = lines.findIndex(line => {
    const match = line.match(/^(#+)\s+(.*)$/);
    return match && match[2].trim() === nodeName.trim();
  });
  if (nodeLineIndex === -1) {
    // Если не нашли, просто добавляем ветку в конец
    return currentMarkdown.trim() + '\n' + branchMarkdown.trim();
  }
  const nodeLevel = (lines[nodeLineIndex].match(/^(#+)/) || ['#'])[0].length;

  // Найти диапазон строк, относящихся к этому узлу (до следующего такого же или меньшего уровня)
  let endIndex = nodeLineIndex + 1;
  while (endIndex < lines.length) {
    const match = lines[endIndex].match(/^(#+)/);
    if (match && match[0].length <= nodeLevel) break;
    endIndex++;
  }

  // branchMarkdown может содержать заголовок nodeName, его нужно заменить на существующий
  // Оставляем только подузлы (всё, что глубже nodeLevel)
  const branchSubLines = branchLines.filter((line, i) => {
    if (i === 0) return false; // пропускаем первый заголовок
    const match = line.match(/^(#+)/);
    return match && match[0].length > nodeLevel;
  });

  // Вставляем новые подузлы вместо старых
  const newLines = [
    ...lines.slice(0, nodeLineIndex + 1),
    ...branchSubLines,
    ...lines.slice(endIndex)
  ];
  return newLines.join('\n').replace(/\n{3,}/g, '\n\n');
}

export const ChatPanel = forwardRef<{ sendMessage: (text: string) => void }, ChatPanelProps>(({
  isOpen,
  onClose,
  onMindMapGenerated,
  onMindMapError,
  onSwitchToMindMap,
  className,
  onTypingChange,
}, ref) => {
  const [isExpanded, setIsExpanded] = useState(false)
  const [isMinimized, setIsMinimized] = useState(false)
  const [mindMapNotification, setMindMapNotification] = useState(false)
  
  // Используем useReducer для управления состоянием
  const [chatState, dispatch] = useReducer(chatReducer, initialState);

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const lastUserMessageRef = useRef<Message | null>(null)

  // Добавляем состояние для отслеживания последнего скролла
  const lastScrollTimeRef = useRef<number>(0);

  // Используем хук useChat из ai/react для управления чатом
  const {
    messages,
    input,
    handleInputChange,
    isLoading,
    stop,
    setMessages,
    append
  } = useChat({
    api: "/api/chat",
    onFinish: async (assistantMessage: Message) => { 
      console.log("Chat finished with assistant message:", assistantMessage);
      
      // Проверяем, что сейчас открыт контекст узла
      const isNodeContext = chatState.currentContext.type === "node";

      // Если последнее сообщение пользователя содержит запрос на расширение узла,
      // определяем это по ID сообщения (вместо регулярного выражения)
      const lastUserMsg = lastUserMessageRef.current; // Используем ref, он должен быть актуален
      const isNodeExpansionQuery = lastUserMsg &&
        lastUserMsg.id.startsWith('node-expand-');

      console.log("[onFinish] Current context type:", chatState.currentContext.type,
        "Node name:", chatState.currentContext.type === "node" ? chatState.currentContext.nodeName : undefined,
        "Is node expansion query (from last msg ID?):", isNodeExpansionQuery,
        "Last user message ID:", lastUserMsg?.id);

      // Если у нас контекст узла или последний запрос был о расширении узла
      if (isNodeContext || isNodeExpansionQuery) {
        const nodeName = chatState.currentContext.type === "node" ? chatState.currentContext.nodeName : '(извлечь из сообщения)';
        let nodeNameToUse = nodeName;

        // Если имя узла не в контексте, пытаемся извлечь из сообщения
        if (nodeNameToUse === '(извлечь из сообщения)' && lastUserMsg) {
            const match = lastUserMsg.content.match(/Расскажи подробнее о \"([^\"]+)\"/);
            if (match && match[1]) {
                nodeNameToUse = match[1];
                console.log(`[onFinish] Node name extracted from message for expansion: ${nodeNameToUse}`);
            } else {
                console.error("[onFinish] Failed to extract node name from expansion query message.");
                return; // Не можем продолжить без имени узла
            }
        }
        
        // Используем assistantMessage как latestMessage для расширения узла
        if (assistantMessage && nodeNameToUse !== '(извлечь из сообщения)') {
            console.log(`[onFinish] Node expansion response received for "${nodeNameToUse}", updating node branch using assistant message ID: ${assistantMessage.id}...`);
            dispatch({ type: "SET_GENERATING", payload: true });
            try {
                await generateMindMap(assistantMessage, { nodeName: nodeNameToUse }, false);
                dispatch({ type: "SET_HAS_MINDMAP", payload: true });
                console.log(`[onFinish] Successfully requested mindmap update for node "${nodeNameToUse}"`);
            } catch (err) {
                console.error("[onFinish] Error requesting mindmap update for node expansion:", err);
            } finally {
                // Флаг генерации сбрасывается внутри generateMindMap
            }
        } else {
            console.warn("[onFinish] Skipping node expansion mindmap update due to missing assistant message or node name.");
        }
      }
      // Если у нас уже есть mindmap и это не расширение узла
      else if (chatState.hasMindMap) {
        const latestUserMessage = lastUserMsg || messages.findLast(m => m.role === 'user');
        if (latestUserMessage) {
            console.log(`[onFinish] Chat response received for message ID ${latestUserMessage.id}, updating full mindmap with new content...`);
            dispatch({ type: "SET_GENERATING", payload: true });
            try {
                await generateMindMap(latestUserMessage, undefined, true);
                console.log("[onFinish] Successfully requested full mindmap update after chat response");
            } catch (err) {
                console.error("[onFinish] Error requesting full mindmap update after chat response:", err);
            } finally {
                // Флаг генерации сбрасывается внутри generateMindMap
            }
        } else {
            console.warn("[onFinish] Skipping full mindmap update because no last user message found.");
        }
        } else {
          // Случай, когда карты еще нет и это не расширение узла
          console.log("[onFinish] No mind map exists yet and not a node expansion. Setting notification.");
          setMindMapNotification(true);
      }
    },
    onError: (error) => {
      console.error("Chat error:", error);
      dispatch({ type: "SET_ERROR", payload: { error: `Ошибка чата: ${error.message || "Неизвестная ошибка"}` } });
    },
    body: {
      mindMapMarkdown: chatState.currentContext.type === "full" ? chatState.currentMindMap : chatState.currentContext.type === "node" ? chatState.currentContext.nodeContext : undefined,
      contextType: chatState.currentContext.type,
      nodeName: chatState.currentContext.type === "node" ? chatState.currentContext.nodeName : undefined
    }
  })

  // Новая функция для генерации предложений по расширению на основе текущей mind map
  const generateExpansionSuggestionsWrapper = (markdown: string) => {
    const suggestions = generateExpansionSuggestions(markdown);
    dispatch({ type: "SET_SUGGESTIONS", payload: suggestions });
  };

  // Обновляем функцию генерации mind map для лучшей поддержки режима узла
  const generateMindMap = async (userMessage: Message, nodeContext?: { nodeName: string }, useAllMessages?: boolean) => {
    if (!userMessage) {
        console.log("generateMindMap: Received empty user message, skipping.");
        dispatch({ type: "SET_GENERATING", payload: false }); 
        return;
    }

    // Логируем ID сообщения
    console.log(`generateMindMap: Processing message with ID: ${userMessage.id}, isNodeExpansion: ${!!nodeContext}`);

    // Определяем, это генерация полной карты или расширение узла
    const isNodeExpansion = !!nodeContext;
    
    console.log(`generateMindMap: Starting ${isNodeExpansion ? 'node expansion' : 'full map'} generation with message:`, userMessage);
    dispatch({ type: "SET_ERROR", payload: { error: null, details: null } });

    // Устанавливаем текущий контекст
    if (isNodeExpansion) {
      console.log(`Generating context for node: ${nodeContext.nodeName}`);
      // Если это расширение узла, используем текущий узел как контекст
      const extractedContext = extractNodeContext(chatState.currentMindMap || '', nodeContext.nodeName);
      dispatch({ type: "SET_CONTEXT", payload: {
        type: "node",
        nodeName: nodeContext.nodeName,
        nodeContext: extractedContext || undefined // заменяем null на undefined
      } });
    } else {
      // Если это полная генерация, используем весь mind map
      dispatch({ type: "SET_CONTEXT", payload: { type: "full" } });
    }

    // Подготовка контекста сообщений для API
    let messageContext = '';
    
    // Если нужно использовать всю историю сообщений
    if (useAllMessages && messages.length > 0) {
      // Создаем контекст из всей истории сообщений
      messageContext = messages.map(msg => 
        `${msg.role === 'user' ? 'Пользователь' : 'Ассистент'}: ${msg.content}`
      ).join('\n\n');
      
      console.log('Generating mindmap from all messages context:', messageContext);
    }

    // Упрощаем и структурируем запрос к API
    const requestParams = {
        // Информация о запросе
        requestType: isNodeExpansion ? 'node_expansion' : 'full_map',
        
        // Основной контент
        latestMessage: { 
            id: userMessage.id,
            role: userMessage.role,
            content: useAllMessages && messageContext ? messageContext : userMessage.content
        },
        
        // Текущее состояние mind map (сохраняем имя поля для совместимости)
        currentMindMapMarkdown: chatState.currentMindMap, 
        
        // Дополнительные параметры
        complexityLevel: chatState.complexityLevel,
        
        // Параметры для расширения узла (только если это расширение)
        nodeContext: isNodeExpansion ? nodeContext : undefined,
        
    };

    try {
      console.log("generateMindMap: Fetching /api/mindmap with params:", requestParams);
      const response = await fetch("/api/mindmap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestParams),
      });
      
      console.log("generateMindMap: Received response status:", response.status);

      if (!response.ok) {
        let errorData = { error: `HTTP error! status: ${response.status}` };
        try {
            errorData = await response.json();
            console.error("generateMindMap: Error response data:", errorData);
        } catch (e) {
            console.error("generateMindMap: Could not parse error response JSON");
        }
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("generateMindMap: Received data:", data);

      if (data.error) {
        console.error("Mind map generation error from API:", data);
        throw new Error(data.error || "Ошибка при генерации mind map от API");
      }

      // Check for mindMapMarkdown field in the response
      if (data.mindMapMarkdown && typeof data.mindMapMarkdown === 'string') {
        console.log("Mind map generated/updated successfully (Markdown)");
        let newMindMapMarkdown = data.mindMapMarkdown;
        if (isNodeExpansion && nodeContext && nodeContext.nodeName) {
          // Встраиваем ветку в текущий mindmap
          newMindMapMarkdown = mergeBranch(chatState.currentMindMap || '', nodeContext.nodeName, data.mindMapMarkdown);
        }
        dispatch({ type: "SET_MINDMAP", payload: newMindMapMarkdown });
        onMindMapGenerated({ mindMapMarkdown: newMindMapMarkdown });
        dispatch({ type: "SET_HAS_MINDMAP", payload: true });
        dispatch({ type: "SET_ERROR", payload: { error: null, details: null } });
        generateExpansionSuggestionsWrapper(newMindMapMarkdown);
      } else {
        console.warn("Mind map API returned success but no mindMapMarkdown field");
        throw new Error("API вернул успешный ответ, но без данных mind map (ожидался Markdown)");
      }
    } catch (error: any) {
      console.error("generateMindMap: CATCH block error:", error);
      const errorMessage = `Ошибка при генерации mind map: ${error.message || "Неизвестная ошибка"}`;
      if (onMindMapError) {
        // Pass the updated requestParams type
        onMindMapError(errorMessage, requestParams);
      } else {
         dispatch({ type: "SET_ERROR", payload: { error: errorMessage } });
      }
      // Не прячем уведомление об ошибке чата, если оно было
    } finally {
      console.log("generateMindMap: Finished generation attempt.");
      dispatch({ type: "SET_GENERATING", payload: false });
    }
  };

  // Обработчик отправки формы
  const handleFormSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const messageContent = input;
    if (!messageContent.trim()) return;

    // Проверяем, является ли это новой темой
    const isStartingNewTopic = isNewTopic(messageContent, chatState.currentMindMap);
    
    // Если это новая тема, сбрасываем текущий mind map и контекст
    if (isStartingNewTopic) {
      console.log("Starting new topic, resetting mind map and context");
      dispatch({ type: "SET_MINDMAP", payload: null });
      dispatch({ type: "SET_CONTEXT", payload: { type: "full" } });
      dispatch({ type: "SET_HAS_MINDMAP", payload: false });
    }

    // Используем чёткий префикс user- для обычных сообщений пользователя
    // чтобы они отличались от сообщений расширения узла (node-expand-)
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: messageContent,
    };
    
    // Очищаем ввод ПЕРЕД отправкой
    handleInputChange({ target: { value: "" } } as any);

    // Сохраняем сообщение пользователя в ref для возможного использования позже
    lastUserMessageRef.current = userMessage;

    dispatch({ type: "SET_ERROR", payload: { error: null, details: null } });
    
    // Отправляем сообщение в чат с явным ID
    // Обновление mindmap произойдет автоматически в callback onFinish
    await append({
      id: userMessage.id,
      role: 'user',
      content: messageContent
    }, {
      data: {
        mindMapMarkdown: chatState.currentContext.type === "full" ? chatState.currentMindMap || undefined : chatState.currentContext.type === "node" ? chatState.currentContext.nodeContext : undefined,
        contextType: chatState.currentContext.type,
        nodeName: chatState.currentContext.type === "node" ? chatState.currentContext.nodeName : undefined,
        // Добавляем системные инструкции в зависимости от типа запроса
        systemInstructions: isStartingNewTopic
          ? `Ты эксперт по созданию образовательного контента. Пользователь запрашивает информацию о новой теме: "${messageContent}".
Ответь на вопрос пользователя и создай хорошо структурированную mind map по этой теме.
Для mind map:
1. Используй научно корректную информацию
2. Структурируй контент с главной темой и 3-7 ключевыми подтемами
3. К каждой подтеме добавь 2-4 важных аспекта
4. Используй краткие и информативные формулировки
${chatState.complexityLevel ? `5. Адаптируй сложность материала под ${chatState.complexityLevel} уровень` : ''}`
          : `Продолжи диалог по текущей теме. Сохрани структуру существующей mind map и при необходимости дополни её новой информацией из ответа.`
      } as any
    });
  };

  // Обновляем функцию для обработки расширения узла
  const handleNodeExpand = (nodeName: string) => {
    if (!nodeName.trim()) return;
    
    // Формируем запрос на расширение узла с четкими инструкциями для LLM
    const expandQuery = `Расскажи подробнее о "${nodeName}"`;
    
    // Если чат еще не открыт, открываем его
    if (!isOpen) {
      onClose(); // Это переключит состояние
    }
    
    // Очищаем предыдущие ошибки
    dispatch({ type: "SET_ERROR", payload: { error: null, details: null } });
    
    // Извлекаем контекст узла
    const nodeContextData = extractNodeContext(chatState.currentMindMap || '', nodeName);
    
    // Устанавливаем правильный тип контекста для этого узла
    dispatch({ type: "SET_CONTEXT", payload: {
      type: "node", 
      nodeName: nodeName,
      nodeContext: nodeContextData || undefined
    } });
    
    console.log(`Expanding node: "${nodeName}", sending query to chat...`);
    console.log("Node context:", nodeContextData);
    
    // Используем уникальный ID с префиксом node-expand-
    const nodeExpandId = `node-expand-${Date.now()}`;
    
    // Запоминаем, что сейчас обрабатываем расширение узла
    lastUserMessageRef.current = {
      id: nodeExpandId,
      role: "user",
      content: expandQuery
    };
    
    // Отправляем запрос в чат с уникальным ID и дальше обновление карты будет после получения ответа в onFinish
    append({
      id: nodeExpandId,
      role: 'user',
      content: expandQuery
    }, {
      data: {
        mindMapMarkdown: chatState.currentMindMap,
        contextType: "node",
        nodeName: nodeName,
        specificHeading: nodeName,
        useNodeContext: true,
        // Добавляем дополнительные инструкции для LLM
        systemInstructions: `Ты эксперт по созданию образовательного контента. Сфокусируйся на узле "${nodeName}". 
Предоставь детальное объяснение этой концепции, сохраняя общую структуру карты. 
Добавь 3-5 ключевых подпунктов и раскрой их. Используй научный, но доступный язык.`
      } as any
    });
  };

  // Фокус на поле ввода при открытии чата
  useEffect(() => {
    if (isOpen && !isLoading && inputRef.current) {
      inputRef.current.focus()
    }
  }, [isOpen, isLoading])

  // Оптимизируем хук для скролла при обновлении сообщений
  useEffect(() => {
    // Используем RAF для оптимизации производительности
    let rafId: number;
    
    const scrollToBottom = () => {
      const now = Date.now();
      
      // Если последнее сообщение от LLM и оно активно генерируется,
      // ограничиваем частоту скролла для повышения производительности
      const isGeneratingAssistantMessage = isLoading && 
          messages.length > 0 && 
          messages[messages.length - 1].role === 'assistant';
      
      const scrollDelay = isGeneratingAssistantMessage ? 500 : 100;
      
      if (now - lastScrollTimeRef.current < scrollDelay) {
        // Отменяем предыдущий requestAnimationFrame, если он еще не выполнен
        if (rafId) cancelAnimationFrame(rafId);
        
        // Планируем новый скролл с задержкой
        rafId = requestAnimationFrame(() => {
          setTimeout(() => {
            performScroll();
          }, scrollDelay);
        });
      } else {
        // Иначе скроллим сразу через requestAnimationFrame
        rafId = requestAnimationFrame(performScroll);
      }
    };
    
    function performScroll() {
      if (!messagesEndRef.current) return;
      
      // Проверяем, находится ли скролл уже близко к низу
      const chatContainer = messagesEndRef.current.parentElement;
      if (!chatContainer) return;
      
      const isNearBottom = 
        chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight < 150;
      
      // Используем плавную прокрутку только если:
      // 1. Скролл был уже близко к низу (пользователь смотрел последние сообщения)
      // 2. Или это первое сообщение (нет необходимости в плавной прокрутке)
      // 3. Или было добавлено новое сообщение пользователя (не обновление существующего от ассистента)
      const isNewUserMessage = messages.length > 0 && messages[messages.length - 1].role === 'user';
      
      if (isNearBottom || messages.length <= 1 || isNewUserMessage) {
        // Используем встроенный скролл вместо scrollIntoView для лучшей производительности
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        // Обновляем время последнего скролла
        lastScrollTimeRef.current = Date.now();
      }
    }
    
    scrollToBottom();
    
    // Очистка при размонтировании
    return () => {
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [messages, isLoading]);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      const form = e.currentTarget.form
      if (form) form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }))
    }
  }

  const stopGeneration = () => {
    stop()
  }

  // Функция для очистки ошибок
  const clearErrors = () => {
    dispatch({ type: "SET_ERROR", payload: { error: null, details: null } });
  }

  // Функция для обновления mind map
  const refreshMindMap = () => {
    const latestUserMessage = messages.findLast(m => m.role === 'user');
    if (latestUserMessage) {
      console.log("Refreshing mind map (Markmap) based on chat history...");
      dispatch({ type: "SET_GENERATING", payload: true });
      
      // Создаем новый объект сообщения с префиксом refresh-
      const refreshMessage: Message = {
        id: `refresh-${Date.now()}`,
        role: "user",
        content: latestUserMessage.content
      };
      
      // Добавляем системные инструкции для регенерации
      const systemInstructions = `Регенерируй mind map на основе всего предшествующего диалога. 
Сохрани основную структуру, но улучши ясность и организацию.
${chatState.complexityLevel ? `Используй уровень сложности: ${chatState.complexityLevel}.` : ''}
Создай хорошо сбалансированную mind map с 5-7 основными ветвями и 2-4 подветвями для каждой.`;
      
      // Используем все сообщения для обновления карты знаний
      generateMindMap(refreshMessage, undefined, true);
    } else {
      console.warn("Cannot refresh mind map, no user messages found.");
      // Maybe set error in parent via onMindMapError?
      dispatch({ type: "SET_ERROR", payload: { error: "Не могу обновить карту, сообщения пользователя не найдены." } });
    }
  };
  
  // Функция для первичной генерации mindmap по запросу пользователя
  const handleGenerateMindmap = async () => {
    if (messages.length === 0) {
      dispatch({ type: "SET_ERROR", payload: { error: "Нет сообщений для генерации карты знаний" } });
      return;
    }
    
    // Используем последнее сообщение пользователя в качестве точки входа
    // и используем весь контекст чата
    const lastUserMessage = messages.findLast(m => m.role === 'user');
    
    if (!lastUserMessage) {
      dispatch({ type: "SET_ERROR", payload: { error: "Не удалось найти сообщение пользователя для генерации карты" } });
      return;
    }
    
    dispatch({ type: "SET_GENERATING", payload: true });
    
    try {
      // Передаем флаг useAllMessages=true для использования всей истории чата
      await generateMindMap(lastUserMessage, undefined, true);
      // После успешной генерации показываем уведомление
      dispatch({ type: "SET_HAS_MINDMAP", payload: true });
      // Автоматически переключаемся на вкладку с картой
      onSwitchToMindMap();
    } catch (err) {
      dispatch({ type: "SET_GENERATING", payload: false });
      dispatch({ type: "SET_ERROR", payload: { error: "Ошибка при генерации карты знаний" } });
    }
  };

  // Слушаем событие expandNode
  useEffect(() => {
    const handleExpandNode = (e: CustomEvent) => {
      if (e.detail?.query) {
        // Извлекаем название узла из запроса
        const nodeNameMatch = e.detail.query.match(/расскажи подробнее о "([^"]+)"/i);
        const nodeName = nodeNameMatch ? nodeNameMatch[1] : null;
        
        if (nodeName) {
          // Вызываем обработчик расширения узла
          handleNodeExpand(nodeName);
        } else {
          // Если не удалось извлечь имя узла, используем стандартный подход
          if (inputRef.current) {
            inputRef.current.value = e.detail.query;
            handleInputChange({ target: { value: e.detail.query } } as any);
            setTimeout(() => {
              // Отправляем форму после обновления значения
              const form = inputRef.current?.form;
              if (form) form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
            }, 10);
          }
        }
      }
    };
    
    document.addEventListener('expandNode', handleExpandNode as EventListener);
    return () => {
      document.removeEventListener('expandNode', handleExpandNode as EventListener);
    };
  }, [handleInputChange]);

  // Функция для очистки чата и mind map
  const clearAll = () => {
    // Очищаем сообщения чата
    setMessages([]);
    // Очищаем текущую mind map
    dispatch({ type: "SET_MINDMAP", payload: null });
    // Сбрасываем контекст
    dispatch({ type: "SET_CONTEXT", payload: { type: "full" } });
    // Сбрасываем состояние генерации
    dispatch({ type: "SET_GENERATING", payload: false });
    // Сбрасываем прогресс
    dispatch({ type: "SET_HAS_MINDMAP", payload: false });
    // Очищаем ошибки
    clearErrors();
    // Передаем пустую mind map в родительский компонент
    onMindMapGenerated({ mindMapMarkdown: "" });
  }

  // Публичный метод для отправки сообщения из родителя
  React.useImperativeHandle(ref, () => ({
    sendMessage: (text: string) => {
      if (inputRef.current) {
        handleInputChange({ target: { value: text } } as any);
        setTimeout(() => {
          if (inputRef.current) {
            const event = new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true });
            inputRef.current.dispatchEvent(event);
          }
        }, 50);
      }
    }
  }), [handleInputChange]);

  // Сообщаем родителю о статусе "печатает" (isLoading или генерация карты)
  useEffect(() => {
    if (onTypingChange) {
      onTypingChange(isLoading || chatState.isGeneratingMindMap);
    }
  }, [isLoading, chatState.isGeneratingMindMap, onTypingChange]);

  return (
    <>
      <div
        className={cn(
          "fixed right-4 bg-background border border-border rounded-t-lg shadow-lg transform transition-all duration-300 ease-in-out z-40",
          isOpen ? "translate-y-0" : "translate-y-full",
          isMinimized
            ? "bottom-0 w-[300px] h-[48px]"
            : isExpanded
              ? "bottom-0 w-[500px] h-[80vh]"
              : "bottom-0 w-[500px] h-[500px]",
          className,
        )}
      >
        <div
          className={cn(
            "flex items-center justify-between p-3 border-b border-border cursor-pointer",
            isMinimized ? "border-b-0" : "",
          )}
          onClick={() => (isMinimized ? setIsMinimized(false) : null)}
        >
          <div className="flex items-center gap-2">
            <Avatar className="h-6 w-6">
              <AvatarFallback>ИИ</AvatarFallback>
            </Avatar>
            <h3 className="font-medium text-sm">Чат с ИИ</h3>
          </div>
          <div className="flex items-center gap-1">
            {!isMinimized && (
              <>
                <TooltipProvider delayDuration={100}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button variant="ghost" size="icon" className="h-7 w-7" onClick={clearAll}>
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Очистить чат и карту</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setIsExpanded(!isExpanded)}>
                  {isExpanded ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
                </Button>
                <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setIsMinimized(true)}>
                  <ChevronDown className="h-3.5 w-3.5" />
                </Button>
              </>
            )}
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onClose}>
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>

        {!isMinimized && (
          <div className="flex flex-col h-[calc(100%-48px)]">
            <div className="flex-1 overflow-y-auto p-4 space-y-4 scroll-smooth scrollbar-thin scrollbar-thumb-rounded scrollbar-thumb-primary/40 scrollbar-track-transparent">
              {messages.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full text-center p-4">
                  <MapIcon className="h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="text-lg font-medium mb-2">Начните общение</h3>
                  <p className="text-muted-foreground mb-4">
                    Задайте вопрос о научной концепции, и я создам для вас интерактивную карту знаний.
                  </p>
                  <div className="flex flex-wrap justify-center gap-2 max-w-xs">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        const value = "Что такое белок?";
                        handleInputChange({ target: { value } } as any);
                        setTimeout(() => {
                          const form = inputRef.current?.form;
                          if (form) form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
                        }, 10);
                      }}
                    >
                      Что такое белок?
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        const value = "Расскажи про ДНК";
                        handleInputChange({ target: { value } } as any);
                        setTimeout(() => {
                          const form = inputRef.current?.form;
                          if (form) form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
                        }, 10);
                      }}
                    >
                      Расскажи про ДНК
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        const value = "Объясни строение атома";
                        handleInputChange({ target: { value } } as any);
                        setTimeout(() => {
                          const form = inputRef.current?.form;
                          if (form) form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
                        }, 10);
                      }}
                    >
                      Строение атома
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        const value = "Как устроена клетка?";
                        handleInputChange({ target: { value } } as any);
                        setTimeout(() => {
                          const form = inputRef.current?.form;
                          if (form) form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
                        }, 10);
                      }}
                    >
                      Как устроена клетка?
                    </Button>
                  </div>
                </div>
              )}

              {chatState.error && (
                <div className="bg-destructive/10 border border-destructive/30 rounded-md p-3 text-sm">
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-2">
                      <AlertCircle className="h-5 w-5 text-destructive mt-0.5" />
                      <div>
                        <p className="font-medium text-destructive mb-1">Ошибка</p>
                        <p className="text-destructive/90">{chatState.error}</p>

                        {chatState.errorDetails && (
                          <div className="mt-2 p-2 bg-destructive/5 rounded border border-destructive/20 overflow-x-auto">
                            <pre className="text-xs text-destructive/80 whitespace-pre-wrap">{chatState.errorDetails}</pre>
                          </div>
                        )}
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 text-destructive/70 hover:text-destructive"
                      onClick={clearErrors}
                    >
                      <X className="h-3.5 w-3.5" />
                    </Button>
                  </div>

                  <div className="mt-3 flex gap-2">
                    <Button variant="outline" size="sm" className="text-xs" onClick={() => setIsExpanded(!isExpanded)}>
                      Проверить режим демонстрации
                    </Button>
                    <Button variant="outline" size="sm" className="text-xs" onClick={clearErrors}>
                      Закрыть
                    </Button>
                  </div>
                </div>
              )}

              {/* Используем виртуализацию для оптимизации отображения большого количества сообщений */}
              <div className="pb-1">
                {messages
                  .filter((msg) => msg.role !== "system")
                  .map((message) => (
                    <div key={message.id} className="mb-4">
                      <ChatMessageItem message={message} />
                    </div>
                  ))}
              </div>

              {chatState.isGeneratingMindMap && (
                <div className="flex mb-4 justify-start">
                  <div className="flex max-w-[80%] flex-row">
                    <Avatar className="h-8 w-8 mt-1">
                      <AvatarFallback className="bg-secondary/10 text-secondary">ИИ</AvatarFallback>
                    </Avatar>
                    <div className="mx-2">
                      <div className="px-3 py-2 rounded-lg bg-muted">
                        <div className="flex space-x-1">
                          <div className="h-2 w-2 bg-foreground/40 rounded-full" style={{ animationDelay: "0ms" }}></div>
                          <div className="h-2 w-2 bg-foreground/40 rounded-full" style={{ animationDelay: "150ms" }}></div>
                          <div className="h-2 w-2 bg-foreground/40 rounded-full" style={{ animationDelay: "300ms" }}></div>
                        </div>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">Генерируется карта знаний...</p>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Заменяем блок с кнопкой Посмотреть Mind Map и обновлением на кнопку генерации */}
            {messages.length > 0 && !chatState.isGeneratingMindMap && !isLoading && (
              <div className="p-3 border-t border-border bg-primary/5">
                <div className="flex items-center gap-2">
                  {chatState.hasMindMap ? (
                      <Button
                        variant="outline"
                        className="flex-1 flex items-center justify-center gap-2 bg-background hover:bg-primary/10"
                        onClick={onSwitchToMindMap}
                      >
                        <MapIcon className="h-4 w-4" />
                        <span>Посмотреть Mind Map</span>
                      </Button>
                  ) : (
                    <Button
                      variant="default"
                      className="flex-1 flex items-center justify-center gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
                      onClick={handleGenerateMindmap}
                    >
                      <MapIcon className="h-4 w-4" />
                      <span>Сгенерировать карту знаний</span>
                    </Button>
                  )}
                </div>
              </div>
            )}

            {/* Поле ввода и кнопка отправки */}
            <div className="p-3 border-t border-border">
              <form onSubmit={handleFormSubmit} className="flex items-center gap-2">
                <Input
                  ref={inputRef}
                  placeholder="Напишите сообщение..."
                  className="flex-1"
                  value={input}
                  onChange={handleInputChange}
                  onKeyDown={handleKeyDown}
                  disabled={isLoading}
                />
                
                {isLoading ? (
                  <Button variant="destructive" size="icon" onClick={stopGeneration} title="Остановить генерацию">
                    <StopCircle className="h-4 w-4" />
                  </Button>
                ) : (
                  <Button type="submit" size="icon" disabled={isLoading}>
                    <Send className="h-4 w-4" />
                  </Button>
                )}
              </form>
              
              {/* НОВЫЙ БЛОК ДЛЯ ИНДИКАТОРА ГЕНЕРАЦИИ MIND MAP */} 
              {chatState.isGeneratingMindMap && (
                <div className="mt-2 px-1 py-1">
                  <div className="text-xs text-muted-foreground flex items-center mb-1">
                    <MapIcon className="h-3.5 w-3.5 mr-1.5 animate-pulse text-primary" />
                    <span>Создание/обновление карты знаний...</span>
                  </div>
                  {/* Используем компонент Progress для визуализации */} 
                  {/* Можно добавить value={progress}, если будем считать прогресс */}
                  <div className="w-full bg-muted rounded-full h-1.5 dark:bg-zinc-700 overflow-hidden">
                     <div className="bg-primary h-1.5 rounded-full animate-pulse" style={{width: "100%"}}></div>
                  </div>
                </div>
              )}
              
              {/* Select под формой */}
              <div className="mt-2 flex items-center gap-2">
                <div className="w-48">
                  <Select value={chatState.complexityLevel} onValueChange={(value) => dispatch({ type: "SET_COMPLEXITY", payload: value })}>
                    <SelectTrigger className="h-7 w-full text-xs">
                      <SelectValue placeholder="Уровень сложности" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="школьный">Школьный уровень</SelectItem>
                      <SelectItem value="университетский">Университетский уровень</SelectItem>
                      <SelectItem value="продвинутый">Продвинутый уровень</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Выпадающий список с вариантами расширения */}
                {chatState.hasMindMap && !chatState.isGeneratingMindMap && chatState.expansionSuggestions.length > 0 && (
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="h-7 text-xs"
                      >
                        Варианты расширения
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent 
                      className="w-auto p-2 mt-1" 
                      align="end" 
                      side="top"
                    >
                      <div className="flex flex-col gap-1 min-w-[200px]">
                        {chatState.expansionSuggestions.map((suggestion, index) => (
                          <Button 
                            key={index}
                            variant="ghost" 
                            size="sm" 
                            className="text-xs justify-start h-7 px-2"
                            onClick={() => {
                              if (inputRef.current) {
                                inputRef.current.value = suggestion;
                                handleInputChange({ target: { value: suggestion } } as any);
                              }
                            }}
                          >
                            {suggestion}
                          </Button>
                        ))}
                      </div>
                    </PopoverContent>
                  </Popover>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  )
})

// Добавляем displayName для лучшей отладки
ChatPanel.displayName = "ChatPanel"
