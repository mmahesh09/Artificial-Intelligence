export default function Console({ output = '' }) {
    return (
      <div className="border rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-2">Console Output</h2>
        <pre className="bg-black text-white p-4 rounded font-mono text-sm">
          {output || 'No output yet...'}
        </pre>
      </div>
    );
  }