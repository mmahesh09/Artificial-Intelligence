import PythonCodeVisualizer from '../components/PythonCodeVisualizer'

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-100 py-8">
      <div className="container mx-auto px-4">
        <h1 className="text-3xl font-bold mb-4">Python Code Visualizer with Enhancer</h1>
        <p className="mb-4">
          Enter your Python code in the text area below. You can use basic Python syntax, including variable assignments and print statements. 
          Click "Run" to execute the code step-by-step, or use the "Step" button to go through the execution manually.
        </p>
        <p className="mb-4">
          New feature: Use the "Show Enhancer" button to get suggestions for improving your code. The enhancer will provide tips on 
          code structure, naming conventions, and best practices.
        </p>
        <PythonCodeVisualizer />
      </div>
    </main>
  )
}

