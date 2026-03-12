import { streamText } from "ai"
import type { MindMapNode } from "@/components/mindmap"
import { OpenAI } from "openai"

// Add type for the expected message structure
interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

// Промпт для первичной генерации структуры mindmap
const initialSystemPrompt = `Ты — ассистент, который преобразует образовательный текст в структуру майндмэпа (mind map).

Твоя задача — проанализировать предоставленный текст ассистента и создать структурированную иерархию в формате Markdown, которая отражает основные понятия и идеи из текста.

Правила:
 • Один заголовок первого уровня (#) — это основная тема текста.
 • заголовки второго уровня (##) — ключевые подтемы, упомянутые в тексте.
 • Заголовков второго уровня, не должно быть более 3-4 штук.
 • Не создавай заголовков третьего и более глубокого уровня.
 • Извлекай ТОЛЬКО информацию, которая присутствует в тексте.
 • Используй только markdown-заголовки (#, ##).
 • Не добавляй списки, обычный текст, комментарии или код.
 • Верни только итоговую структуру — без дополнительных пояснений.

ВАЖНО: Не добавляй никакой информации, которой нет в тексте ассистента.`;

// Промпт для расширения/углубления существующей структуры
const expansionSystemPrompt = `Ты — ассистент, который преобразует образовательный текст в структуру майндмэпа (mind map).

Твоя задача — обновить существующий майндмэп, ТОЛЬКО на 1 уровень вглубь.

Правила:
 • Если текст содержит новую информацию о существующем узле, добавь эту информацию как подузлы.
 • Используй логичные уровни иерархии: ##, ###, ####.
 • Новых добавляемых узлов не должно быть более 3-4 штук.
 • Новые добавляемые узлы должны быть на одном уровне глубины майндмэпа.
 • ИСПОЛЬЗУЙ ТОЛЬКО информацию из предоставленного текста.
 • Используй только markdown-заголовки (#, ##, ###, ####).
 • Не добавляй обычный текст, комментарии или код.
 • Верни ПОЛНУЮ обновленную структуру.
 Если ты считаешь, что в тексте нет новой информации, верни исходный майндмэп.

ВАЖНО: Твоя задача —добавить новый уровень в майндмэп. Не додумывай и не добавляй информацию, которой нет в тексте ассистента.`;

export const maxDuration = 300

export async function POST(req: Request) {
  console.log("Markmap API: Received POST request.");
  try {
    // Expect latestMessage, currentMindMapMarkdown, and complexityLevel
    const { latestMessage, currentMindMapMarkdown, complexityLevel, nodeContext }: { 
      latestMessage?: ChatMessage, 
      currentMindMapMarkdown?: string,
      complexityLevel?: "школьный" | "университетский" | "продвинутый",
      nodeContext?: { nodeName: string }
    } = await req.json()
    
    console.log("Markmap API received latestMessage:", !!latestMessage, 
               "Has currentMindMapMarkdown:", !!currentMindMapMarkdown,
               "Complexity level:", complexityLevel || "default",
               "Node context:", nodeContext?.nodeName || "none");

    // Validate latestMessage
    if (!latestMessage || typeof latestMessage !== 'object' || !latestMessage.role || !latestMessage.content) {
      console.error("Markmap API: Invalid or missing latestMessage object");
      return new Response(JSON.stringify({ error: "Latest message object is required" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      })
    }

    const apiKey = process.env.OPENAI_API_KEY
    if (!apiKey) {
      console.error("Markmap API: OpenAI API key not found.");
      return new Response(JSON.stringify({ error: "OpenAI API ключ не настроен." }), {
        status: 500,
        headers: { "Content-Type": "application/json" }
      });
    }
    console.log("Markmap API: Found API Key.");

    // Выбираем промпт в зависимости от типа запроса
    // Если есть nodeContext или есть текущий mindmap, используем промпт для расширения
    // Иначе используем промпт для первичной генерации
    const systemPrompt = (nodeContext || currentMindMapMarkdown) 
      ? expansionSystemPrompt 
      : initialSystemPrompt;

    // Construct user prompt with complexity level if provided
    let userPromptContent: string;
    
    const complexityNote = complexityLevel 
      ? `\n\nУровень сложности: ${complexityLevel}`
      : "";
      
    if (currentMindMapMarkdown) {
      // Если это расширение узла, указываем на какой узел нужно сфокусироваться
      const nodeInstruction = nodeContext 
        ? `\n\nПользователь кликнул на узел "${nodeContext.nodeName}". Расширь этот узел, добавив подузлы на основе текста ассистента.` 
        : "";
        
      userPromptContent = `Текущая mind map (Markdown):\n\`\`\`markdown\n${currentMindMapMarkdown}\n\`\`\`\n\nТекст ассистента для преобразования в структуру:\n${latestMessage.content}${complexityNote}${nodeInstruction}\n\nПреобразуй текст ассистента в markdown-структуру, обновив существующую mind map. Верни ТОЛЬКО полный обновленный Markdown mind map.`;
    } else {
      userPromptContent = `Текст ассистента для преобразования в структуру:\n${latestMessage.content}${complexityNote}\n\nПреобразуй этот текст в markdown-структуру для mind map. Верни ТОЛЬКО Markdown mind map.`;
    }
      
    const augmentedMessages = [
      { role: "system", content: systemPrompt },
      {
        role: "user",
        content: userPromptContent,
      },
    ];
    
    console.log("Markmap API: Prepared messages for OpenAI.");
    
    // Инициализируем клиент OpenAI с OpenRouter
    const openai = new OpenAI({ 
      apiKey: apiKey,
      baseURL: "https://openrouter.ai/api/v1"
    });

    console.log("Markmap API: Sending request to OpenAI via OpenRouter...");
    let completion;
    try {
      completion = await openai.chat.completions.create({
        model: "gpt-4o", // Используем доступную модель в OpenRouter с правильным форматом
        messages: augmentedMessages as any,
        temperature: 0.1, // Lower for more predictable Markdown
        max_tokens: 2000, // Limit response size
      });
      console.log("Markmap API: Received response from OpenRouter.");
    } catch (openaiError: any) {
        console.error("Markmap API: Error during OpenRouter request:", openaiError);
        throw new Error(`OpenRouter API request failed: ${openaiError.message}`);
    }

    const content = completion?.choices?.[0]?.message?.content?.trim() || "";
    if (!content) {
        console.error("Markmap API: OpenAI response content is empty.");
        throw new Error("OpenAI response content is empty.");
    }
    console.log("Markmap API: Received Markdown content length:", content.length);

    // Очистка Markdown
    const cleanMarkdown = (markdown: string | null | undefined): string | null => {
      if (!markdown) return null;
      // Удаляем обрамляющие тройные кавычки и возможный тег markdown
      return markdown.replace(/^```(markdown)?\n/m, '').replace(/\n```$/m, '');
    };
    const cleanedResponse = cleanMarkdown(content) || content;
    console.log("Markmap API: Returning cleaned Markdown response.");

    // --- ФИЛЬТРАЦИЯ ВЕТКИ ДЛЯ nodeContext ---
    let finalResponse = cleanedResponse;
    if (nodeContext?.nodeName) {
      const lines = finalResponse.split("\n");
      const branch = [];
      let inBranch = false;
      for (const line of lines) {
        if (line.trim() === `## ${nodeContext.nodeName}`) {
          inBranch = true;
          continue;
        }
        if (!inBranch) continue;
        // собираем все заголовки глубже ##
        if (/^###/.test(line) || /^####/.test(line) || /^#####/.test(line)) {
          branch.push(line);
        } else {
          // как только встретили следующий ## — выходим
          if (/^## [^#]/.test(line)) break;
        }
      }
      if (branch.length) {
        finalResponse = branch.join("\n");
      }
    }

    // Return cleaned Markdown as plain text or in JSON wrapper
    return new Response(JSON.stringify({ mindMapMarkdown: finalResponse }), { 
      headers: { "Content-Type": "application/json" } 
    });

  } catch (error: any) {
    console.error("Markmap API: CATCH block error:", error);
    return new Response(
      JSON.stringify({
        error: "Internal server error in Markmap API",
        message: error.message || "Unknown error",
      }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
