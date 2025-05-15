//utils.ts
function paramsToObject(entries) {
  const result = {}
  // each 'entry' is a [key, value] tupple
  for (const [key, value] of entries) {
    result[key] = value;
  }
  return result;
}

export { paramsToObject }