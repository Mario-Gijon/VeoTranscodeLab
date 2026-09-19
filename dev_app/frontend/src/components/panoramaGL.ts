import type { ViewContext } from '../types'

/** Feed the captured shaders unchanged. Python supplies column-major uniforms,
 * including the validated inverse camera2presentation convention. */
export function createPanoramaGL(canvas: HTMLCanvasElement, context: ViewContext) {
  const gl = canvas.getContext('webgl', { alpha: false, preserveDrawingBuffer: true })
  if (!gl) throw new Error('WebGL is unavailable. Use the CPU frame render below.')
  const compile = (type: number, source: string) => {
    const shader = gl.createShader(type)!
    gl.shaderSource(shader, source); gl.compileShader(shader)
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) || 'Shader compilation failed')
    return shader
  }
  const vertex = compile(gl.VERTEX_SHADER, context.vertex_shader)
  const fragment = compile(gl.FRAGMENT_SHADER, context.fragment_shader)
  const program = gl.createProgram()!
  gl.attachShader(program, vertex); gl.attachShader(program, fragment); gl.linkProgram(program)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error('Captured panorama shader could not link')
  gl.useProgram(program)
  const buffers: WebGLBuffer[] = []
  const attribute = (name: string, values: number[]) => {
    const buffer = gl.createBuffer()!; buffers.push(buffer)
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(values), gl.STATIC_DRAW)
    const location = gl.getAttribLocation(program, name)
    gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0)
  }
  attribute('a_position', [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1])
  attribute('a_texcoord', [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1])
  Object.entries(context.uniforms).forEach(([name, uniform]) => {
    const location = gl.getUniformLocation(program, name)
    if (uniform.matrix === 4) gl.uniformMatrix4fv(location, false, new Float32Array(uniform.value as number[]))
    else if (uniform.matrix === 3) gl.uniformMatrix3fv(location, false, new Float32Array(uniform.value as number[]))
    else gl.uniform1f(location, uniform.value as number)
  })
  const texture = (unit: number, name: string) => {
    const result = gl.createTexture()!
    gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, result)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.uniform1i(gl.getUniformLocation(program, name), unit)
    return result
  }
  const source = texture(0, 'texture')
  const watermark = texture(1, 'watermarkTexture')
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 0]))
  return {
    draw(image: TexImageSource) {
      gl.viewport(0, 0, canvas.width, canvas.height)
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, source)
      // Source texture y=0 is the TOP of the stacked video in the captured shader.
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image)
      gl.drawArrays(gl.TRIANGLES, 0, 6)
    },
    dispose() {
      buffers.forEach((buffer) => gl.deleteBuffer(buffer))
      gl.deleteTexture(source); gl.deleteTexture(watermark)
      gl.deleteShader(vertex); gl.deleteShader(fragment); gl.deleteProgram(program)
    },
  }
}
