"use client"

import type React from "react"

import { useState } from "react"
import { ChevronDown, ChevronUp, Database } from "lucide-react"
import { useToast } from "@/components/ui/use-toast"
import { Checkbox } from "@/components/ui/checkbox"
import { Button } from "@/components/ui/button"

interface FilterSectionProps {
  title: string
  children: React.ReactNode
  defaultOpen?: boolean
}

const FilterSection = ({ title, children, defaultOpen = false }: FilterSectionProps) => {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className="mb-4">
      <div
        className="flex items-center justify-between text-[#353e5c] cursor-pointer py-1"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span className="text-sm font-medium">{title}</span>
        {isOpen ? <ChevronUp className="w-4 h-4 text-[#727987]" /> : <ChevronDown className="w-4 h-4 text-[#727987]" />}
      </div>

      {isOpen && <div className="mt-2 text-sm text-[#353e5c]">{children}</div>}
    </div>
  )
}

export function SidebarFilter() {
  const { toast } = useToast()
  const [selectedGraph, setSelectedGraph] = useState("biology_graph")
  const [loading, setLoading] = useState(false)

  // Функция для переключения графа
  const switchGraph = async (graphName: string) => {
    if (selectedGraph === graphName) return // Если тот же граф, не делаем ничего
    
    setLoading(true)
    try {
      const response = await fetch('/api/graph/switch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ graph_name: graphName })
      })
      
      const data = await response.json()
      
      if (response.ok && data.success) {
        setSelectedGraph(graphName)
        toast({
          title: "Граф изменен",
          description: `Переключено на граф: ${getGraphDisplayName(graphName)}`,
        })
        
        // Перезагружаем страницу для обновления данных
        window.location.reload()
      } else {
        throw new Error(data.detail || "Ошибка при переключении графа")
      }
    } catch (error) {
      console.error("Ошибка при переключении графа:", error)
      toast({
        title: "Ошибка",
        description: `Не удалось переключить граф: ${error instanceof Error ? error.message : String(error)}`,
        variant: "destructive"
      })
    } finally {
      setLoading(false)
    }
  }

  // Отображаем названия графов в более дружественном формате
  const getGraphDisplayName = (graphName: string) => {
    switch (graphName) {
      case "biology_graph": return "Биология 5 класс"
      case "physics_graph": return "Физика 10 класс"
      case "history_graph": return "История России 11 класс"
      default: return graphName
    }
  }

  return (
    <div className="p-4 border-t border-[#eaeaea]">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-[#353e5c]">ФИЛЬТРЫ</h3>
          <button className="text-xs text-[#727987] hover:text-[#2388ff]">
            Сбросить
          </button>
        </div>

        <div className="mb-6">
          <h4 className="text-base font-medium mb-3 text-[#353e5c]">Выберите материал</h4>
          
          <div className="space-y-3">
            <Button
              variant={selectedGraph === "biology_graph" ? "default" : "outline"}
              className="w-full justify-start"
              onClick={() => switchGraph("biology_graph")}
              disabled={loading}
            >
              <Database className="mr-2 h-4 w-4" />
              Биология 5 класс
              {loading && selectedGraph === "biology_graph" && <span className="ml-2 animate-spin">⟳</span>}
            </Button>
            
            <Button
              variant={selectedGraph === "physics_graph" ? "default" : "outline"}
              className="w-full justify-start"
              onClick={() => switchGraph("physics_graph")}
              disabled={loading}
            >
              <Database className="mr-2 h-4 w-4" />
              Физика 10 класс
              {loading && selectedGraph === "physics_graph" && <span className="ml-2 animate-spin">⟳</span>}
            </Button>
            
            <Button
              variant={selectedGraph === "history_graph" ? "default" : "outline"}
              className="w-full justify-start"
              onClick={() => switchGraph("history_graph")}
              disabled={loading}
            >
              <Database className="mr-2 h-4 w-4" />
              История России 11 класс
              {loading && selectedGraph === "history_graph" && <span className="ml-2 animate-spin">⟳</span>}
            </Button>
          </div>
        </div>
      </div>

      <FilterSection title="Тема" defaultOpen={true}>
        <div className="py-1 flex items-center">
          <Checkbox id="topic1" className="mr-2" />
          <label htmlFor="topic1" className="text-[#353e5c] cursor-pointer">Алгебра</label>
        </div>
        <div className="py-1 flex items-center">
          <Checkbox id="topic2" className="mr-2" />
          <label htmlFor="topic2" className="text-[#353e5c] cursor-pointer">Геометрия</label>
        </div>
        <div className="py-1 flex items-center">
          <Checkbox id="topic3" className="mr-2" />
          <label htmlFor="topic3" className="text-[#353e5c] cursor-pointer">Механика</label>
        </div>
        <div className="py-1 flex items-center">
          <Checkbox id="topic4" className="mr-2" />
          <label htmlFor="topic4" className="text-[#353e5c] cursor-pointer">Электричество</label>
        </div>
      </FilterSection>
    </div>
  )
}
