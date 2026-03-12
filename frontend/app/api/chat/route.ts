import { openai, createOpenAI } from '@ai-sdk/openai';
import { streamText, CoreMessage } from 'ai';

// Determine runtime environment
export const runtime = 'edge';

export async function POST(req: Request) {
  try {
    // Extract messages and relevant data from request
    const { messages, mindMapMarkdown, contextType, nodeName, specificHeading, systemInstructions, useNodeContext } = await req.json();
    
    // Расширенное логгирование
    console.log(`Chat API: Received ${messages?.length || 0} messages`);
    console.log(`Chat API: Mindmap markdown present: ${!!mindMapMarkdown}`);
    console.log(`Chat API: Context type: ${contextType || 'full'}`);
    console.log(`Chat API: Node name: ${nodeName || 'N/A'}`);
    console.log(`Chat API: Specific heading: ${specificHeading || 'N/A'}`);
    console.log(`Chat API: Has System Instructions: ${!!systemInstructions}`);
    console.log(`Chat API: Use Node Context flag: ${!!useNodeContext}`);
    
    // Определяем и логируем тип контекста (можно упростить, если specificHeading всегда означает node)
    const contextMode = (contextType === 'node' || specificHeading || useNodeContext)
      ? 'node_explanation'
      : 'initial_overview';
    console.log(`Chat API: Using context mode: ${contextMode}`);
    
    // Validate messages
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return new Response(JSON.stringify({ error: "Messages array is required" }), { 
        status: 400,
        headers: { "Content-Type": "application/json" } 
      });
    }

    // Check for OpenAI API key
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return new Response(JSON.stringify({ error: "OpenAI API key not found" }), { 
        status: 500,
        headers: { "Content-Type": "application/json" } 
      });
    }

    // --- ОБНОВЛЕНО: Используем systemInstructions, если они есть --- 
    // Базовые промпты
    const baseInitialPrompt = `Ты — образовательный ассистент, который помогает пользователю изучать новую тему. Твоя задача — дать структурированное и понятное объяснение запрашиваемой темы.
Тебя зовут GigaChat, ты разработан сотрудниками Сбера.
Правила:
 • БУДЬ СУПЕР КРАТОК БЕЗ ДЕТАЛЕЙ
 • Раздели тему на 3-4 ключевых аспекта.
 • Каждый аспект объясни в отдельном абзаце.
 • Давай точную, научно корректную информацию.
 • Используй простой, ясный язык.
 • Приводи понятные примеры и аналогии для сложных концепций.
 • Выделяй ключевые термины в тексте полужирным шрифтом (**Термин**).
 • Используй абзацы для разделения смысловых блоков. Отделяй абзацы друг от друга тремя <br>.
 • Не используй нумерованные списки, маркированные списки и другие специальные форматы.`;

    const baseNodeExplanationPrompt = `Ты — образовательный ассистент, который помогает пользователю изучать тему через диалог и визуализацию. Пользователь кликнул на элемент карты знаний и хочет узнать подробнее об этой конкретной подтеме.
Тебя зовут GigaChat, ты разработан сотрудниками Сбера.
Текст который ты напишешь будет использоваться для расширения интерактивной карты знаний.
Правила:
 • БУДЬ СУПЕР КРАТОК БЕЗ ДЕТАЛЕЙ
 • Расширь подтему на 1 уровень вниз.
 • Каждый аспект объясни в отдельном абзаце.
 • Давай точную, научно корректную информацию.
 • Используй простой, ясный язык.
 • Приводи понятные примеры и аналогии для сложных концепций.
 • Выделяй ключевые термины в тексте полужирным шрифтом (**Термин**).
 • Используй абзацы для разделения смысловых блоков. Отделяй абзацы друг от друга тремя <br>.
 • Не используй нумерованные списки, маркированные списки и другие специальные форматы.`;

    let systemPrompt = '';

    // Если переданы явные системные инструкции, используем их
    if (systemInstructions && typeof systemInstructions === 'string') {
        systemPrompt = systemInstructions;
        console.log("Chat API: Using provided system instructions.");
    }
    // Иначе выбираем базовый промпт по контексту
    else if (contextMode === 'node_explanation') {
        systemPrompt = baseNodeExplanationPrompt;
        if (specificHeading) {
            systemPrompt += `\n\nВажно: Пользователь кликнул на узел \"${specificHeading}\". Объясни этот узел и его подтемы, если они есть.`;
        } else if (nodeName) {
            systemPrompt += `\n\nВажно: Пользователь просит рассказать о \"${nodeName}\". Объясни этот узел и его подтемы, если они есть.`;
        }
        console.log("Chat API: Using base node explanation prompt.");
    } else {
        systemPrompt = baseInitialPrompt;
        console.log("Chat API: Using base initial overview prompt.");
    }
    // --- КОНЕЦ ОБНОВЛЕНИЯ --- 

    // Логируем финальный используемый промпт
    console.log(`Chat API: Final system prompt (first 100 chars): ${systemPrompt.substring(0, 100)}...`);

    // Создаем кастомный провайдер OpenAI для OpenRouter
    const openrouterProvider = createOpenAI({
      baseURL: "https://openrouter.ai/api/v1",
      apiKey: apiKey
    });
    
    // Создаем модель с использованием OpenRouter
    const model = openrouterProvider('gpt-4o');

    // Подготовка сообщений для API OpenAI
    const lastUserMessage = messages.filter(m => m.role === 'user').pop();
    
    const augmentedMessages: CoreMessage[] = [
      { role: 'system', content: systemPrompt } as CoreMessage,
      // Передаем ТОЛЬКО последнее сообщение пользователя, чтобы избежать дублирования контекста, который уже в systemPrompt
      // или в mindMapMarkdown
      ...(lastUserMessage ? [{ role: lastUserMessage.role, content: lastUserMessage.content } as CoreMessage] : [])
      // Старый вариант: ...messages.map(...) - может быть слишком многословно
    ];

    // Если есть markdown майндмэпа И это не объяснение узла (где контекст уже специфичен),
    // включаем его в системный промпт или сообщение пользователя.
    // Можно добавить его к systemPrompt или оставить логику добавления к последнему сообщению пользователя.
    // Давайте попробуем добавить к systemPrompt для краткости.
    let finalSystemPrompt = systemPrompt;
    if (mindMapMarkdown && contextMode !== 'node_explanation') {
        finalSystemPrompt += `\n\nТекущая структура майндмэпа:\n\`\`\`markdown\n${mindMapMarkdown}\n\`\`\``;
        console.log("Chat API: Appended mindMapMarkdown to system prompt for full overview.");
    } else if (mindMapMarkdown && contextMode === 'node_explanation') {
        // Для объяснения узла, возможно, лучше передать markdown в user message?
        // Или положиться на specificHeading/nodeName в systemPrompt?
        // Пока оставим как есть - не добавляем markdown к systemPrompt для node_explanation.
        console.log("Chat API: MindMap markdown present but not appended to system prompt for node explanation.");
    }

    // Обновляем системное сообщение в augmentedMessages
    if (augmentedMessages[0]?.role === 'system') {
        augmentedMessages[0].content = finalSystemPrompt;
    }
    
    // Убираем дублирование markdown из последнего сообщения пользователя, если оно там было
    const lastUserIndex = augmentedMessages.findIndex(msg => msg.role === 'user');
    if (lastUserIndex !== -1) {
        const userContent = augmentedMessages[lastUserIndex].content;
        if (typeof userContent === 'string' && userContent.includes('Вот структура майндмэпа:')) {
            augmentedMessages[lastUserIndex].content = userContent.split('\n\nВот структура майндмэпа:')[0];
            console.log("Chat API: Cleaned mindmap markdown structure from last user message to avoid duplication.");
        }
    }

    console.log(`Chat API: Sending ${augmentedMessages.length} messages to LLM (System + Last User)`);
    console.log(`Chat API: Final System Prompt (start): ${finalSystemPrompt.substring(0, 200)}...`);
    const userMessageContentForLog = augmentedMessages[1]?.content;
    if (augmentedMessages[1]) console.log(`Chat API: User Message: ${typeof userMessageContentForLog === 'string' ? userMessageContentForLog.substring(0, 200) : '(non-string content)'}...`);

    // Используем streamText с AI SDK
    const result = await streamText({
      model: model,
      messages: augmentedMessages,
      temperature: 0.1,
    });

    console.log(`Chat API: Stream initialized successfully`);

    // Преобразуем результат напрямую в потоковый ответ
    return result.toDataStreamResponse();
    
  } catch (error: any) {
    console.error('Chat API error:', error);
    return new Response(JSON.stringify({ error: error.message }), { 
      status: 500,
      headers: { "Content-Type": "application/json" } 
    });
  }
}
