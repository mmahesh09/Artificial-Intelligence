export function formatOutput(output) {
    return output.trim();
  }
  
  export function parseVariables(vars) {
    return Object.entries(vars).reduce((acc, [key, value]) => {
      acc[key] = typeof value === 'object' ? JSON.stringify(value) : value;
      return acc;
    }, {});
  }