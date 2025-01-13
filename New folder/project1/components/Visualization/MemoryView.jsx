export default function MemoryView({ variables = {} }) {
    return (
      <div className="border rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-2">Memory View</h2>
        <div className="space-y-2">
          {Object.entries(variables).map(([name, value]) => (
            <div key={name} className="flex justify-between">
              <span className="font-mono">{name}</span>
              <span className="font-mono">{JSON.stringify(value)}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }