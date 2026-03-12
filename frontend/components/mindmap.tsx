"use client"

import { useRef, useEffect, useState, useCallback } from "react"
import * as d3 from "d3"
import { cn } from "@/lib/utils"
import { NodePreview } from "@/components/node-preview"

export interface MindMapNode {
  id: string
  name: string
  children?: MindMapNode[]
  collapsed?: boolean
  depth?: number
  x?: number
  y?: number
  parent?: MindMapNode
}

interface MindMapProps {
  data: MindMapNode
  width?: number
  height?: number
  className?: string
  onNodeClick?: (node: MindMapNode) => void
}

export function MindMap({ data, width = 800, height = 600, className, onNodeClick }: MindMapProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [hoveredNode, setHoveredNode] = useState<MindMapNode | null>(null)
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 })
  const [transform, setTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity)
  const previewTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const hideTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const isMouseOverPreviewRef = useRef(false)
  const activeNodeRef = useRef<MindMapNode | null>(null)

  // Debounced function to show node preview
  const showNodePreview = useCallback((node: MindMapNode, x: number, y: number) => {
    // Clear any existing timeouts
    if (previewTimeoutRef.current) {
      clearTimeout(previewTimeoutRef.current)
    }
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current)
    }

    // Set position immediately to avoid jumps
    setHoverPosition({ x, y })

    // Store the active node reference
    activeNodeRef.current = node

    // Set a timeout to show the preview
    previewTimeoutRef.current = setTimeout(() => {
      if (activeNodeRef.current === node) {
        setHoveredNode(node)
      }
    }, 600) // Increased delay for more stability
  }, [])

  // Function to hide node preview with delay
  const hideNodePreview = useCallback(() => {
    // Only set hide timeout if mouse is not over preview
    if (!isMouseOverPreviewRef.current) {
      if (hideTimeoutRef.current) {
        clearTimeout(hideTimeoutRef.current)
      }

      hideTimeoutRef.current = setTimeout(() => {
        if (!isMouseOverPreviewRef.current) {
          setHoveredNode(null)
          activeNodeRef.current = null
        }
      }, 300) // Delay before hiding to allow mouse to move to preview
    }
  }, [])

  // Handle mouse entering preview
  const handleMouseEnterPreview = useCallback(() => {
    isMouseOverPreviewRef.current = true
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current)
    }
  }, [])

  // Handle mouse leaving preview
  const handleMouseLeavePreview = useCallback(() => {
    isMouseOverPreviewRef.current = false
    hideNodePreview()
  }, [hideNodePreview])

  // Get color for node based on its type or depth
  const getNodeColor = useCallback((node: MindMapNode, depth: number) => {
    // Define color palette for different node types
    const colors = {
      root: { fill: "hsl(var(--primary))", text: "hsl(var(--primary-foreground))", stroke: "hsl(var(--primary))" },
      level1: { fill: "hsl(var(--primary-light))", text: "hsl(var(--foreground))", stroke: "hsl(var(--border))" },
      level2: { fill: "hsl(var(--secondary-light))", text: "hsl(var(--foreground))", stroke: "hsl(var(--border))" },
      level3: { fill: "hsl(var(--accent-light))", text: "hsl(var(--foreground))", stroke: "hsl(var(--border))" },
      default: { fill: "hsl(var(--card))", text: "hsl(var(--foreground))", stroke: "hsl(var(--border))" },
    }

    // Special nodes by ID or name
    if (node.id === "protein" || node.name?.includes("Белок")) {
      return { fill: "#22c55e20", text: "currentColor", stroke: "#22c55e" }
    }
    if (node.id === "dna" || node.name?.includes("ДНК")) {
      return { fill: "#3b82f620", text: "currentColor", stroke: "#3b82f6" }
    }
    if (node.id === "atom" || node.name?.includes("Атом")) {
      return { fill: "#f59e0b20", text: "currentColor", stroke: "#f59e0b" }
    }
    if (node.id === "cell" || node.name?.includes("Клетка")) {
      return { fill: "#06b6d420", text: "currentColor", stroke: "#06b6d4" }
    }

    // Colors based on depth
    if (depth === 0) return colors.root
    if (depth === 1) return colors.level1
    if (depth === 2) return colors.level2
    if (depth === 3) return colors.level3
    return colors.default
  }, [])

  // Clear all timeouts on unmount
  useEffect(() => {
    return () => {
      if (previewTimeoutRef.current) {
        clearTimeout(previewTimeoutRef.current)
      }
      if (hideTimeoutRef.current) {
        clearTimeout(hideTimeoutRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (!svgRef.current) return

    // Clear any existing SVG content
    d3.select(svgRef.current).selectAll("*").remove()

    const svg = d3.select(svgRef.current)
    const container = svg.append("g")

    // Define gradient for links
    const defs = svg.append("defs")

    // Create gradient for links
    const linkGradient = defs
      .append("linearGradient")
      .attr("id", "linkGradient")
      .attr("gradientUnits", "userSpaceOnUse")

    linkGradient.append("stop").attr("offset", "0%").attr("stop-color", "hsl(var(--primary))").attr("stop-opacity", 0.6)

    linkGradient.append("stop").attr("offset", "100%").attr("stop-color", "hsl(var(--muted))").attr("stop-opacity", 0.3)

    // Process the data to add depth and collapsed state
    const root = d3.hierarchy(data) as d3.HierarchyNode<MindMapNode>

    // Create tree layout
    const treeLayout = d3
      .tree<MindMapNode>()
      .size([height - 100, width - 200])
      .nodeSize([80, 200])
      .separation((a, b) => (a.parent === b.parent ? 1 : 1.5))

    // Apply the layout
    const rootNode = treeLayout(root)

    // Create links with curved paths and animations
    const links = container
      .append("g")
      .attr("class", "links")
      .selectAll("path")
      .data(rootNode.links())
      .enter()
      .append("path")
      .attr("d", (d) => {
        return `M${d.source.y},${d.source.x}
                C${(d.source.y + d.target.y) / 2},${d.source.x}
                 ${(d.source.y + d.target.y) / 2},${d.target.x}
                 ${d.target.y},${d.target.x}`
      })
      .attr("fill", "none")
      .attr("stroke", "url(#linkGradient)")
      .attr("stroke-width", 1.5)
      .attr("stroke-opacity", 0.8)
      .attr("stroke-dasharray", function () {
        const length = this.getTotalLength()
        return `${length} ${length}`
      })
      .attr("stroke-dashoffset", function () {
        return this.getTotalLength()
      })
      .transition()
      .duration(500)
      .delay((d, i) => i * 20)
      .attr("stroke-dashoffset", 0)

    // Create nodes
    const nodes = container
      .append("g")
      .attr("class", "nodes")
      .selectAll("g")
      .data(rootNode.descendants())
      .enter()
      .append("g")
      .attr("transform", (d) => `translate(${d.y},${d.x})`)
      .attr("cursor", "pointer")
      .attr("opacity", 0)
      .transition()
      .duration(500)
      .delay((d, i) => i * 30)
      .attr("opacity", 1)

    // Add invisible larger rectangle for better hover detection
    container
      .selectAll(".node-hitarea")
      .data(rootNode.descendants())
      .enter()
      .append("rect")
      .attr("class", "node-hitarea")
      .attr("x", (d) => d.y - 70)
      .attr("y", (d) => d.x - 30)
      .attr("width", 140)
      .attr("height", 60)
      .attr("rx", 6)
      .attr("ry", 6)
      .attr("fill", "transparent")
      .style("pointer-events", "all")
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        // Clear any hover state
        if (previewTimeoutRef.current) {
          clearTimeout(previewTimeoutRef.current)
        }
        if (hideTimeoutRef.current) {
          clearTimeout(hideTimeoutRef.current)
        }
        
        // Получаем координаты клика
        const [x, y] = d3.pointer(event, svg.node())
        
        // Показываем карточку ноды
        setHoverPosition({ x, y })
        setHoveredNode(d.data)
        activeNodeRef.current = d.data

        // Highlight the node
        d3.select(`.node-rect-${d.data.id}`)
          .transition()
          .duration(200)
          .attr("filter", "url(#glow)")
          .attr("stroke-width", 2)

        event.stopPropagation() // Prevent zoom behavior
      })

    // Create glow filter for hover effect
    const filter = defs
      .append("filter")
      .attr("id", "glow")
      .attr("x", "-50%")
      .attr("y", "-50%")
      .attr("width", "200%")
      .attr("height", "200%")

    filter.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "blur")

    filter.append("feComposite").attr("in", "SourceGraphic").attr("in2", "blur").attr("operator", "over")

    // Add node rectangles with improved styling
    container
      .selectAll(".node-rect")
      .data(rootNode.descendants())
      .enter()
      .append("rect")
      .attr("class", (d) => `node-rect node-rect-${d.data.id}`)
      .attr("x", (d) => d.y - 60)
      .attr("y", (d) => d.x - 20)
      .attr("width", 120)
      .attr("height", 40)
      .attr("rx", 6)
      .attr("ry", 6)
      .attr("fill", (d) => getNodeColor(d.data, d.depth).fill)
      .attr("stroke", (d) => getNodeColor(d.data, d.depth).stroke || "var(--border)")
      .attr("stroke-width", 1)
      .style("filter", (d) => (d.depth === 0 ? "drop-shadow(0 2px 3px rgba(0,0,0,0.1))" : "none"))
      .style("pointer-events", "none") // Let the larger invisible rectangle handle events

    // Add node text with improved styling
    container
      .selectAll(".node-text")
      .data(rootNode.descendants())
      .enter()
      .append("text")
      .attr("class", "node-text text-foreground")
      .attr("x", (d) => d.y)
      .attr("y", (d) => d.x)
      .attr("dy", "0.35em")
      .attr("text-anchor", "middle")
      .attr("font-size", (d) => (d.depth === 0 ? "14px" : "12px"))
      .attr("font-weight", (d) => (d.depth === 0 ? "500" : "400"))
      .style("fill", "currentColor")
      .each(function (d) {
        const text = d3.select(this)
        const words = d.data.name.split(/\s+/)
        const lineHeight = 1.1
        const y = text.attr("y")
        const dy = Number.parseFloat(text.attr("dy"))

        text.text(null)

        if (words.length > 1) {
          const tspan1 = text
            .append("tspan")
            .attr("x", d.y)
            .attr("y", y)
            .attr("dy", `${dy}em`)
            .text(words.slice(0, Math.ceil(words.length / 2)).join(" "))

          const tspan2 = text
            .append("tspan")
            .attr("x", d.y)
            .attr("y", y)
            .attr("dy", `${lineHeight}em`)
            .text(words.slice(Math.ceil(words.length / 2)).join(" "))
        } else {
          text.append("tspan").attr("x", d.y).attr("y", y).attr("dy", `${dy}em`).text(d.data.name)
        }
      })
      .style("pointer-events", "none") // Make text non-interactive to avoid interfering with rect events

    // Add expand/collapse icons for nodes with children
    container
      .selectAll(".toggle-group")
      .data(rootNode.descendants().filter((d) => d.children))
      .enter()
      .append("g")
      .attr("class", "toggle-group")
      .attr("transform", (d) => `translate(${d.y + 70},${d.x})`)
      .append("circle")
      .attr("r", 10)
      .attr("fill", "var(--secondary)")
      .attr("stroke", "var(--border)")
      .attr("stroke-width", 1)
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        // Toggle node expansion logic would go here
        event.stopPropagation()
      })

    container
      .selectAll(".toggle-group")
      .append("text")
      .attr("font-family", "sans-serif")
      .attr("font-size", "14px")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .style("fill", "currentColor")
      .attr("class", "text-foreground")
      .text("-") // Always show the - symbol since we're not collapsing
      .style("pointer-events", "none")

    // Click on background to clear hover state
    svg.on("click", () => {
      if (previewTimeoutRef.current) {
        clearTimeout(previewTimeoutRef.current)
      }
      if (hideTimeoutRef.current) {
        clearTimeout(hideTimeoutRef.current)
      }
      
      // Сбрасываем выделение всех нод
      container.selectAll(".node-rect").attr("filter", null).attr("stroke-width", 1)
      
      setHoveredNode(null)
      activeNodeRef.current = null
    })

    // Set up zoom behavior
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 2])
      .on("zoom", (event) => {
        container.attr("transform", event.transform.toString())
        setTransform(event.transform)
      })

    svg.call(zoom)

    // Center the mindmap
    const initialTransform = d3.zoomIdentity.translate(width / 2 - 100, height / 2).scale(0.8)

    svg.call(zoom.transform, initialTransform)

    // Cleanup
    return () => {
      svg.on(".zoom", null)
    }
  }, [data, width, height, onNodeClick, showNodePreview, hideNodePreview, getNodeColor])

  return (
    <div className={cn("relative w-full h-full", className)}>
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        className="bg-transparent"
      />

      {hoveredNode && (
        <NodePreview
          node={hoveredNode}
          position={hoverPosition}
          onClose={() => {
            // Также сбрасываем выделение ноды при закрытии карточки
            if (hoveredNode && hoveredNode.id) {
              d3.select(`.node-rect-${hoveredNode.id}`).attr("filter", null).attr("stroke-width", 1)
            }
            setHoveredNode(null)
            activeNodeRef.current = null
          }}
          onViewDetails={() => {
            // Только здесь вызываем onNodeClick для открытия детальной панели
            if (onNodeClick) onNodeClick(hoveredNode)
            
            // Сбрасываем выделение ноды
            if (hoveredNode && hoveredNode.id) {
              d3.select(`.node-rect-${hoveredNode.id}`).attr("filter", null).attr("stroke-width", 1)
            }
            
            // Закрываем карточку предпросмотра
            setHoveredNode(null)
            activeNodeRef.current = null
          }}
          onMouseEnter={handleMouseEnterPreview}
          onMouseLeave={handleMouseLeavePreview}
        />
      )}
    </div>
  )
}
