(() => {
  const state = window.VEO_VALIDATION;
  const panorama = document.getElementById("panorama");
  const panoramaCanvas = document.getElementById("panorama-markers");
  const pitchCanvas = document.getElementById("pitch");
  const pointList = document.getElementById("point-list");
  const emptyState = document.getElementById("empty-state");
  const clearButton = document.getElementById("clear-points");
  const points = [];

  function drawLabel(context, text, x, y) {
    context.font = "bold 18px system-ui";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.lineWidth = 4;
    context.strokeStyle = "#ffffff";
    context.strokeText(text, x, y);
    context.fillStyle = "#17211b";
    context.fillText(text, x, y);
  }

  function drawPanoramaMarkers() {
    const context = panoramaCanvas.getContext("2d");
    context.clearRect(0, 0, panoramaCanvas.width, panoramaCanvas.height);
    points.forEach((point, index) => {
      const color = point.status === "inside_field" ? "#f7d23e" : "#f39827";
      context.beginPath();
      context.arc(point.pixel_x + .5, point.pixel_y + .5, 9, 0, Math.PI * 2);
      context.fillStyle = color;
      context.fill();
      context.lineWidth = 2;
      context.strokeStyle = "#17211b";
      context.stroke();
      drawLabel(context, String(index + 1), point.pixel_x + .5, point.pixel_y + .5);
    });
  }

  function pitchTransform(context) {
    const halfLength = state.field_length / 2;
    const halfWidth = state.field_width / 2;
    let minX = -halfLength;
    let maxX = halfLength;
    let minZ = -halfWidth;
    let maxZ = halfWidth;
    points.filter(point => point.world_x !== null).forEach(point => {
      minX = Math.min(minX, point.world_x);
      maxX = Math.max(maxX, point.world_x);
      minZ = Math.min(minZ, point.world_z);
      maxZ = Math.max(maxZ, point.world_z);
    });
    const margin = 5;
    minX -= margin; maxX += margin; minZ -= margin; maxZ += margin;
    const scale = Math.min(context.canvas.width / (maxX - minX), context.canvas.height / (maxZ - minZ));
    const offsetX = (context.canvas.width - (maxX - minX) * scale) / 2 - minX * scale;
    const offsetY = (context.canvas.height - (maxZ - minZ) * scale) / 2 + maxZ * scale;
    return (x, z) => [offsetX + x * scale, offsetY - z * scale];
  }

  function strokeSegment(context, toCanvas, x1, z1, x2, z2, width = 2) {
    const [sx, sy] = toCanvas(x1, z1);
    const [ex, ey] = toCanvas(x2, z2);
    context.lineWidth = width;
    context.beginPath();
    context.moveTo(sx, sy);
    context.lineTo(ex, ey);
    context.stroke();
  }

  function strokeRectangle(context, toCanvas, minX, minZ, maxX, maxZ) {
    strokeSegment(context, toCanvas, minX, minZ, maxX, minZ);
    strokeSegment(context, toCanvas, maxX, minZ, maxX, maxZ);
    strokeSegment(context, toCanvas, maxX, maxZ, minX, maxZ);
    strokeSegment(context, toCanvas, minX, maxZ, minX, minZ);
  }

  function drawPitch() {
    const context = pitchCanvas.getContext("2d");
    context.clearRect(0, 0, pitchCanvas.width, pitchCanvas.height);
    context.fillStyle = "#1f713e";
    context.fillRect(0, 0, pitchCanvas.width, pitchCanvas.height);
    const toCanvas = pitchTransform(context);
    const halfLength = state.field_length / 2;
    const halfWidth = state.field_width / 2;
    context.strokeStyle = "#ffffff";
    context.fillStyle = "#ffffff";
    context.lineCap = "round";
    strokeRectangle(context, toCanvas, -halfLength, -halfWidth, halfLength, halfWidth);
    strokeSegment(context, toCanvas, 0, -halfWidth, 0, halfWidth);

    const [centerX, centerY] = toCanvas(0, 0);
    const [circleEdgeX] = toCanvas(9.15, 0);
    context.beginPath();
    context.arc(centerX, centerY, Math.abs(circleEdgeX - centerX), 0, Math.PI * 2);
    context.stroke();
    context.beginPath();
    context.arc(centerX, centerY, 3, 0, Math.PI * 2);
    context.fill();

    strokeRectangle(context, toCanvas, -halfLength, -20.16, -halfLength + 16.5, 20.16);
    strokeRectangle(context, toCanvas, halfLength - 16.5, -20.16, halfLength, 20.16);
    strokeRectangle(context, toCanvas, -halfLength, -9.16, -halfLength + 5.5, 9.16);
    strokeRectangle(context, toCanvas, halfLength - 5.5, -9.16, halfLength, 9.16);
    [[-halfLength + 11, 0], [halfLength - 11, 0]].forEach(([x, z]) => {
      const [px, py] = toCanvas(x, z);
      context.beginPath(); context.arc(px, py, 3, 0, Math.PI * 2); context.fill();
    });

    points.forEach((point, index) => {
      if (point.world_x === null) return;
      const [x, y] = toCanvas(point.world_x, point.world_z);
      context.beginPath();
      context.arc(x, y, 10, 0, Math.PI * 2);
      context.fillStyle = point.status === "inside_field" ? "#f7d23e" : "#f39827";
      context.fill();
      context.lineWidth = 2;
      context.strokeStyle = "#17211b";
      context.stroke();
      drawLabel(context, String(index + 1), x, y);
    });
  }

  function statusText(point) {
    if (point.status === "inside_field") return "Inside field";
    if (point.status === "outside_field") return "Outside field";
    return "No ground intersection";
  }

  function renderPointList() {
    pointList.replaceChildren();
    emptyState.hidden = points.length !== 0;
    points.forEach((point, index) => {
      const item = document.createElement("li");
      const status = document.createElement("span");
      status.className = point.status === "inside_field" ? "inside" : point.status === "outside_field" ? "outside" : "no-ground";
      status.textContent = statusText(point);
      item.append(`#${index + 1} pixel (${point.pixel_x}, ${point.pixel_y}) · UV (${point.u.toFixed(7)}, ${point.v.toFixed(7)}) · `);
      if (point.world_x !== null) {
        item.append(`X=${point.world_x.toFixed(3)} m · Z=${point.world_z.toFixed(3)} m · `);
      }
      item.append(status);
      pointList.append(item);
    });
  }

  function redraw() {
    drawPanoramaMarkers();
    drawPitch();
    renderPointList();
  }

  panoramaCanvas.addEventListener("click", async event => {
    const bounds = panoramaCanvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(state.width - 1, Math.floor((event.clientX - bounds.left) * state.width / bounds.width)));
    const y = Math.max(0, Math.min(state.height - 1, Math.floor((event.clientY - bounds.top) * state.height / bounds.height)));
    const response = await fetch("/api/click", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({x, y}),
    });
    const point = await response.json();
    if (response.ok) {
      points.push(point);
      redraw();
    } else {
      window.alert(point.error || "Could not calculate the ground intersection.");
    }
  });

  clearButton.addEventListener("click", () => {
    points.length = 0;
    redraw();
  });

  panorama.addEventListener("load", redraw);
  redraw();
})();
