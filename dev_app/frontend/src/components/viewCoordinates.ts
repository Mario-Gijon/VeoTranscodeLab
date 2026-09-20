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

/** Maps a click in a possibly letterboxed video element to native video pixels. */
export function nativeVideoPixel(clientX: number, clientY: number, bounds: { left: number; top: number; width: number; height: number }, nativeWidth: number, nativeHeight: number) {
  const scale = Math.min(bounds.width / nativeWidth, bounds.height / nativeHeight)
  const drawnWidth = nativeWidth * scale
  const drawnHeight = nativeHeight * scale
  const offsetX = (bounds.width - drawnWidth) / 2
  const offsetY = (bounds.height - drawnHeight) / 2
  return { x: Math.max(0, Math.min(nativeWidth - 1, Math.floor((clientX - bounds.left - offsetX) / scale))), y: Math.max(0, Math.min(nativeHeight - 1, Math.floor((clientY - bounds.top - offsetY) / scale))) }
}
