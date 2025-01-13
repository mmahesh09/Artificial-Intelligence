import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Lightbulb } from 'lucide-react'

const CodeEnhancer = ({ code }) => {
  const suggestions = generateSuggestions(code)

  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle className="flex items-center">
          <Lightbulb className="mr-2 h-5 w-5" />
          Code Enhancement Suggestions
        </CardTitle>
      </CardHeader>
      <CardContent>
        {suggestions.length > 0 ? (
          <ul className="space-y-2">
            {suggestions.map((suggestion, index) => (
              <li key={index} className="flex items-start">
                <Badge variant="outline" className="mr-2 mt-1">{suggestion.type}</Badge>
                <span>{suggestion.message}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p>No suggestions at this time. Your code looks good!</p>
        )}
      </CardContent>
    </Card>
  )
}

const generateSuggestions = (code) => {
  const suggestions = []
  const lines = code.split('\n')

  lines.forEach((line, index) => {
    // Check for overly complex expressions
    if (line.split('=').length > 2) {
      suggestions.push({
        type: 'Simplify',
        message: `Line ${index + 1}: Consider breaking down complex expressions into multiple lines for better readability.`
      })
    }

    // Check for single-letter variable names
    const singleLetterVars = line.match(/\b[a-z]\b/g)
    if (singleLetterVars && singleLetterVars.length > 0) {
      suggestions.push({
        type: 'Naming',
        message: `Line ${index + 1}: Consider using more descriptive variable names instead of single letters.`
      })
    }

    // Check for hardcoded values
    if (line.match(/\b\d+\b/) && !line.startsWith('print')) {
      suggestions.push({
        type: 'Constants',
        message: `Line ${index + 1}: Consider using constants for hardcoded values to improve maintainability.`
      })
    }

    // Check for multiple spaces (inconsistent indentation)
    if (line.match(/\s{2,}/)) {
      suggestions.push({
        type: 'Style',
        message: `Line ${index + 1}: Use consistent indentation (preferably 4 spaces per level).`
      })
    }
  })

  // Check overall code structure
  if (!code.includes('def ')) {
    suggestions.push({
      type: 'Structure',
      message: 'Consider organizing your code into functions for better modularity and reusability.'
    })
  }

  return suggestions
}

export default CodeEnhancer

