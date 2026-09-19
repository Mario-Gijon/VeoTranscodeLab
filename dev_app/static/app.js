(() => {
  const state = window.VEO_VALIDATION;
  const panorama = document.getElementById("panorama");
  const panoramaCanvas = document.getElementById("panorama-markers");
  const pitchCanvas = document.getElementById("pitch");
  const pointList = document.getElementById("point-list");
  const pointTable = document.getElementById("point-table");
  const emptyState = document.getElementById("empty-state");
  const clearButton = document.getElementById("clear-points");
  const clickMode = document.getElementById("click-mode");
  const landmarkPicker = document.getElementById("landmark-picker");
  const landmarkSelect = document.getElementById("landmark-select");
  const magnifier = document.getElementById("magnifier");
  const magnificationLabel = document.getElementById("magnification-label");
  const points = [];
  let magnification = 4;
  let lastPointer = null;

  function cssToPanoramaPixel(event) {
    const bounds = panoramaCanvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(state.width - 1, Math.floor((event.clientX - bounds.left) * state.width / bounds.width)));
    const y = Math.max(0, Math.min(state.height - 1, Math.floor((event.clientY - bounds.top) * state.height / bounds.height)));
    return {x, y, cssX: event.clientX - bounds.left, cssY: event.clientY - bounds.top};
  }

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
    points.forEach(point => {
      if (point.measured_x !== null) {
        minX = Math.min(minX, point.measured_x);
        maxX = Math.max(maxX, point.measured_x);
        minZ = Math.min(minZ, point.measured_z);
        maxZ = Math.max(maxZ, point.measured_z);
      }
      if (point.expected_x !== null) {
        minX = Math.min(minX, point.expected_x);
        maxX = Math.max(maxX, point.expected_x);
        minZ = Math.min(minZ, point.expected_z);
        maxZ = Math.max(maxZ, point.expected_z);
      }
    });
    const margin = 5;
    minX -= margin; maxX += margin; minZ -= margin; maxZ += margin;
    const scale = Math.min(context.canvas.width / (maxX - minX), context.canvas.height / (maxZ - minZ));
    const offsetX = (context.canvas.width - (maxX - minX) * scale) / 2 - minX * scale;
    const offsetY = (context.canvas.height - (maxZ - minZ) * scale) / 2 + maxZ * scale;
    return (x, z) => [offsetX + x * scale, offsetY - z * scale];
  }

  function strokeSegment(context, toCanvas, x1, z1, x2, z2, width = 2) {
    const start = toCanvas(x1, z1);
    const end = toCanvas(x2, z2);
    context.lineWidth = width;
    context.beginPath();
    context.moveTo(start[0], start[1]);
    context.lineTo(end[0], end[1]);
    context.stroke();
  }

  function strokeRectangle(context, toCanvas, minX, minZ, maxX, maxZ) {
    strokeSegment(context, toCanvas, minX, minZ, maxX, minZ);
    strokeSegment(context, toCanvas, maxX, minZ, maxX, maxZ);
    strokeSegment(context, toCanvas, maxX, maxZ, minX, maxZ);
    strokeSegment(context, toCanvas, minX, maxZ, minX, minZ);
  }

  function drawExpectedMarker(context, x, y, index) {
    context.save();
    context.translate(x, y);
    context.rotate(Math.PI / 4);
    context.fillStyle = "#45b5e9";
    context.strokeStyle = "#17211b";
    context.lineWidth = 2;
    context.fillRect(-8, -8, 16, 16);
    context.strokeRect(-8, -8, 16, 16);
    context.restore();
    drawLabel(context, String(index + 1), x, y);
  }

  function drawMeasuredMarker(context, x, y, point, index) {
    context.beginPath();
    context.arc(x, y, 10, 0, Math.PI * 2);
    context.fillStyle = point.status === "inside_field" ? "#f7d23e" : "#f39827";
    context.fill();
    context.lineWidth = 2;
    context.strokeStyle = "#17211b";
    context.stroke();
    drawLabel(context, String(index + 1), x, y);
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

    const center = toCanvas(0, 0);
    const circleEdge = toCanvas(9.15, 0);
    context.beginPath();
    context.arc(center[0], center[1], Math.abs(circleEdge[0] - center[0]), 0, Math.PI * 2);
    context.stroke();
    context.beginPath();
    context.arc(center[0], center[1], 3, 0, Math.PI * 2);
    context.fill();

    strokeRectangle(context, toCanvas, -halfLength, -20.16, -halfLength + 16.5, 20.16);
    strokeRectangle(context, toCanvas, halfLength - 16.5, -20.16, halfLength, 20.16);
    strokeRectangle(context, toCanvas, -halfLength, -9.16, -halfLength + 5.5, 9.16);
    strokeRectangle(context, toCanvas, halfLength - 5.5, -9.16, halfLength, 9.16);
    [[-halfLength + 11, 0], [halfLength - 11, 0]].forEach(position => {
      const pixel = toCanvas(position[0], position[1]);
      context.beginPath(); context.arc(pixel[0], pixel[1], 3, 0, Math.PI * 2); context.fill();
    });

    points.forEach((point, index) => {
      if (point.expected_x !== null && point.measured_x !== null) {
        context.strokeStyle = "#45b5e9";
        strokeSegment(context, toCanvas, point.expected_x, point.expected_z, point.measured_x, point.measured_z, 1.5);
      }
      if (point.expected_x !== null) {
        const expected = toCanvas(point.expected_x, point.expected_z);
        drawExpectedMarker(context, expected[0], expected[1], index);
      }
      if (point.measured_x !== null) {
        const measured = toCanvas(point.measured_x, point.measured_z);
        drawMeasuredMarker(context, measured[0], measured[1], point, index);
      }
    });
  }

  function statusText(point) {
    if (point.status === "inside_field") return "Inside field";
    if (point.status === "outside_field") return "Outside field";
    return "No ground intersection";
  }

  function coordinatePair(x, z) {
    return x === null ? "—" : x.toFixed(3) + ", " + z.toFixed(3) + " m";
  }

  function cell(text) {
    const item = document.createElement("td");
    item.textContent = text;
    return item;
  }

  function renderPointList() {
    pointList.replaceChildren();
    emptyState.hidden = points.length !== 0;
    pointTable.hidden = points.length === 0;
    points.forEach((point, index) => {
      const row = document.createElement("tr");
      row.append(
        cell(String(index + 1)),
        cell(point.landmark_name || "Free point"),
        cell("(" + point.pixel_x + ", " + point.pixel_y + ") / (" + point.u.toFixed(7) + ", " + point.v.toFixed(7) + ")"),
        cell(coordinatePair(point.expected_x, point.expected_z)),
        cell(coordinatePair(point.measured_x, point.measured_z)),
        cell(point.delta_x === null ? "—" : point.delta_x.toFixed(3) + ", " + point.delta_z.toFixed(3) + " m"),
        cell(point.field_error_m === null ? "—" : point.field_error_m.toFixed(3) + " m"),
      );
      const status = document.createElement("td");
      status.className = point.status === "inside_field" ? "inside" : point.status === "outside_field" ? "outside" : "no-ground";
      status.textContent = statusText(point);
      row.append(status);
      pointList.append(row);
    });
  }

  function updateMagnifier(pointer) {
    if (pointer === null) return;
    const size = 180;
    const width = panoramaCanvas.clientWidth;
    const height = panoramaCanvas.clientHeight;
    let left = pointer.cssX + 18;
    let top = pointer.cssY + 18;
    if (left + size > width) left = pointer.cssX - size - 18;
    if (top + size > height) top = pointer.cssY - size - 18;
    left = Math.max(0, Math.min(left, Math.max(0, width - size)));
    top = Math.max(0, Math.min(top, Math.max(0, height - size)));
    magnifier.style.left = left + "px";
    magnifier.style.top = top + "px";
    magnifier.style.backgroundImage = "url('" + (panorama.currentSrc || panorama.src) + "')";
    magnifier.style.backgroundSize = (state.width * magnification) + "px " + (state.height * magnification) + "px";
    magnifier.style.backgroundPosition = (size / 2 - (pointer.x + .5) * magnification) + "px " + (size / 2 - (pointer.y + .5) * magnification) + "px";
    magnificationLabel.textContent = magnification + "×";
    magnifier.style.display = "block";
  }

  function redraw() {
    drawPanoramaMarkers();
    drawPitch();
    renderPointList();
  }

  clickMode.addEventListener("change", () => {
    landmarkPicker.hidden = clickMode.value !== "landmark";
  });

  panoramaCanvas.addEventListener("mousemove", event => {
    lastPointer = cssToPanoramaPixel(event);
    updateMagnifier(lastPointer);
  });

  panoramaCanvas.addEventListener("mouseleave", () => {
    lastPointer = null;
    magnifier.style.display = "none";
  });

  panoramaCanvas.addEventListener("wheel", event => {
    event.preventDefault();
    magnification = Math.max(2, Math.min(10, magnification + (event.deltaY < 0 ? 1 : -1)));
    if (lastPointer !== null) updateMagnifier(lastPointer);
  }, {passive: false});

  panoramaCanvas.addEventListener("click", async event => {
    const pixel = cssToPanoramaPixel(event);
    const landmark = clickMode.value === "landmark" ? landmarkSelect.value : null;
    const response = await fetch("/api/click", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({x: pixel.x, y: pixel.y, landmark}),
    });
    const point = await response.json();
    if (response.ok) {
      point.point_index = points.length + 1;
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

  panorama.addEventListener("load", () => {
    if (lastPointer !== null) updateMagnifier(lastPointer);
    redraw();
  });
  redraw();
})();
