/** Display geometry only. Metric camera geometry lives in Python. */
export function naturalPixel(clientX: number, clientY: number, bounds: { left: number; top: number; width: number; height: number }, width: number, height: number) {
  return {
    x: Math.max(0, Math.min(width - 1, Math.floor((clientX - bounds.left) * width / bounds.width))),
    y: Math.max(0, Math.min(height - 1, Math.floor((clientY - bounds.top) * height / bounds.height))),
  }
}

export function stackedPixel(x: number, y: number, crop: string) {
  return { x, y: y + (crop === 'right' ? 1024 : 0) }
}
