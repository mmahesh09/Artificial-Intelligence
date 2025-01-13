"use client"

import React, { useState, useEffect, useRef } from 'react'
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle, Play, Pause, SkipForward, RotateCcw, Zap } from 'lucide-react'
import Prism from 'prismjs'
import 'prismjs/components/prism-python'
import 'prismjs/themes/prism.css'
import CodeEnhancer from './CodeEnhancer'

const PythonCodeVisualizer = () => {
  const [code, setCode] = useState('')
  const [executionSteps, setExecutionSteps] = useState([])
  const [currentStep, setCurrentStep] = useState(-1)
  const [isRunning, setIsRunning] = useState(false)
  const [speed, setSpeed] = useState(1000) // milliseconds between steps
  const [showEnhancer, setShowEnhancer] = useState(false)
  const runIntervalRef = useRef(null)

  useEffect(() => {
    Prism.highlightAll()
  }, [code, executionSteps])

  const executeCode = () => {
    const lines = code.split('\n')
    const steps = []
    let variables = {}
    let output = ''

    const safeEval = (code, context) => {
      const contextKeys = Object.keys(context)
      const contextValues = Object.values(context)
      return new Function(...contextKeys, `return ${code}`)(...contextValues)
    }

    lines.forEach((line, index) => {
      const trimmedLine = line.trim()
      if (trimmedLine) {
        try {
          if (trimmedLine.startsWith('print')) {
            const printContent = trimmedLine.slice(6, -1)
            const evaluatedContent = safeEval(printContent, variables)
            output += evaluatedContent + '\n'
          } else if (trimmedLine.includes('=')) {
            const [varName, varValue] = trimmedLine.split('=').map(part => part.trim())
            variables[varName] = safeEval(varValue, variables)
          } else {
            safeEval(trimmedLine, variables)
          }

          steps.push({
            lineNumber: index + 1,
            code: trimmedLine,
            variables: { ...variables },
            output: output,
            error: null
          })
        } catch (error) {
          console.error(`Error executing line ${index + 1}: ${error}`)
          output += `Error: ${error}\n`
          steps.push({
            lineNumber: index + 1,
            code: trimmedLine,
            variables: { ...variables },
            output: output,
            error: error.toString()
          })
        }
      }
    })

    setExecutionSteps(steps)
    setCurrentStep(-1)
  }

  const runCode = () => {
    if (currentStep === -1) {
      executeCode()
    }
    setIsRunning(true)
    runIntervalRef.current = setInterval(() => {
      setCurrentStep(prevStep => {
        if (prevStep + 1 >= executionSteps.length) {
          clearInterval(runIntervalRef.current)
          setIsRunning(false)
          return prevStep
        }
        return prevStep + 1
      })
    }, speed)
  }

  const pauseCode = () => {
    clearInterval(runIntervalRef.current)
    setIsRunning(false)
  }

  const stepForward = () => {
    if (currentStep === -1) {
      executeCode()
    }
    setCurrentStep(prevStep => Math.min(prevStep + 1, executionSteps.length - 1))
  }

  const resetCode = () => {
    clearInterval(runIntervalRef.current)
    setIsRunning(false)
    setCurrentStep(-1)
    setExecutionSteps([])
  }

  const toggleEnhancer = () => {
    setShowEnhancer(!showEnhancer)
  }

  return (
    <div className="container mx-auto p-4">
      <Card className="mb-4">
        <CardHeader>
          <CardTitle>Python Code Visualizer</CardTitle>
        </CardHeader>
        <CardContent>
          <Textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            placeholder="Enter your Python code here..."
            className="min-h-[200px] font-mono"
          />
        </CardContent>
        <CardFooter className="flex justify-between">
          <div>
            <Button onClick={isRunning ? pauseCode : runCode} className="mr-2">
              {isRunning ? <Pause className="mr-2 h-4 w-4" /> : <Play className="mr-2 h-4 w-4" />}
              {isRunning ? 'Pause' : 'Run'}
            </Button>
            <Button onClick={stepForward} disabled={isRunning} className="mr-2">
              <SkipForward className="mr-2 h-4 w-4" />
              Step
            </Button>
            <Button onClick={toggleEnhancer} variant="outline">
              <Zap className="mr-2 h-4 w-4" />
              {showEnhancer ? 'Hide' : 'Show'} Enhancer
            </Button>
          </div>
          <Button onClick={resetCode} variant="outline">
            <RotateCcw className="mr-2 h-4 w-4" />
            Reset
          </Button>
        </CardFooter>
      </Card>

      {showEnhancer && <CodeEnhancer code={code} />}

      {executionSteps.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Execution Steps</CardTitle>
          </CardHeader>
          <CardContent>
            {executionSteps.map((step, index) => (
              <div key={index} className={`mb-4 p-4 border rounded ${index === currentStep ? 'bg-yellow-100' : ''}`}>
                <h3 className="font-bold">Step {index + 1} (Line {step.lineNumber})</h3>
                <pre className="language-python">
                  <code>{step.code}</code>
                </pre>
                <h4 className="font-semibold mt-2">Variables:</h4>
                <pre className="bg-gray-100 p-2 rounded">
                  {JSON.stringify(step.variables, null, 2)}
                </pre>
                {step.output && (
                  <>
                    <h4 className="font-semibold mt-2">Output:</h4>
                    <pre className={`p-2 rounded ${step.error ? 'bg-red-100' : 'bg-gray-100'}`}>
                      {step.output}
                    </pre>
                  </>
                )}
                {step.error && (
                  <Alert variant="destructive" className="mt-2">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>
                      {step.error}
                    </AlertDescription>
                  </Alert>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export default PythonCodeVisualizer

