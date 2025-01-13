export default function Timeline({ steps = [] }) {
    return (
      <div className="border rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-2">Execution Timeline</h2>
        <div className="space-y-2">
          {steps.map((step, index) => (
            <div key={index} className="flex items-center space-x-2">
              <span className="font-mono">{index + 1}</span>
              <span>{step}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }