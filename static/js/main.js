// ============================================================
// BREAST CANCER ANALYSIS - MAIN JAVASCRIPT
// ============================================================

document.addEventListener("DOMContentLoaded", function () {
  // ============================================================
  // ELEMENT REFERENCES
  // ============================================================

  const uploadArea = document.getElementById("uploadArea");
  const fileInput = document.getElementById("fileInput");
  const previewArea = document.getElementById("previewArea");
  const previewImage = document.getElementById("previewImage");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const resetBtn = document.getElementById("resetBtn");
  const resultsSection = document.getElementById("results");
  const loadingOverlay = document.getElementById("loadingOverlay");

  let selectedFile = null;

  // ============================================================
  // FILE UPLOAD HANDLING
  // ============================================================

  // Click to upload
  uploadArea.addEventListener("click", () => fileInput.click());

  // Drag and drop
  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
  });

  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
  });

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  });

  // File input change
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  // Handle selected file
  function handleFile(file) {
    if (!file.type.match("image.*")) {
      alert("Please select an image file (JPG or PNG)");
      return;
    }

    if (file.size > 16 * 1024 * 1024) {
      alert("File size must be less than 16MB");
      return;
    }

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      uploadArea.style.display = "none";
      previewArea.style.display = "block";
    };
    reader.readAsDataURL(file);
  }

  // Reset button
  resetBtn.addEventListener("click", () => {
    selectedFile = null;
    previewImage.src = "";
    fileInput.value = "";
    uploadArea.style.display = "block";
    previewArea.style.display = "none";
    resultsSection.style.display = "none";
  });

  // ============================================================
  // ANALYZE IMAGE
  // ============================================================

  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) {
      alert("Please select an image first");
      return;
    }

    // Show loading
    loadingOverlay.style.display = "flex";

    // Prepare form data
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.error) {
        alert("Error: " + data.error);
        return;
      }

      displayResults(data);
    } catch (error) {
      alert("Error analyzing image: " + error.message);
    } finally {
      loadingOverlay.style.display = "none";
    }
  });

  // ============================================================
  // DISPLAY RESULTS
  // ============================================================

  function displayResults(data) {
    resultsSection.style.display = "block";

    // Prediction badge
    const badge = document.getElementById("predictionBadge");
    badge.textContent = data.prediction.class;
    badge.className =
      "prediction-badge " + data.prediction.class.split(" ")[0].toLowerCase();

    // Confidence
    document.getElementById("confidenceValue").textContent =
      data.prediction.confidence.toFixed(1) + "%";

    // Model used
    document.getElementById("modelUsedValue").textContent =
      data.metrics.model_used || "HoVerNet-GNN (Hybrid)";

    // Metrics
    document.getElementById("nucleiCount").textContent =
      data.metrics.num_nuclei;
    document.getElementById("tumorPercentage").textContent =
      data.metrics.tumor_percentage.toFixed(1) + "%";
    document.getElementById("cellDensity").textContent =
      data.metrics.cell_density;
    document.getElementById("tumorGrade").textContent =
      data.metrics.tumor_grade;
    document.getElementById("numConnections").textContent =
      data.metrics.num_connections;

    // Images
    document.getElementById("imgOriginal").src =
      "data:image/png;base64," + data.images.original;
    document.getElementById("imgSegmentation").src =
      "data:image/png;base64," + data.images.segmentation;
    document.getElementById("imgHeatmap").src =
      "data:image/png;base64," + data.images.heatmap;
    document.getElementById("imgDensity").src =
      "data:image/png;base64," + data.images.density_map;
    document.getElementById("imgXai").src =
      "data:image/png;base64," + data.images.xai_overlay;
    document.getElementById("imgSpatial").src =
      "data:image/png;base64," + data.images.spatial_graph;
    document.getElementById("imgHv").src =
      "data:image/png;base64," + data.images.hv_gradient;

    // XAI Explanation
    const xaiBox = document.getElementById("xaiExplanation");
    if (data.xai_explanation) {
      xaiBox.textContent = data.xai_explanation;
      xaiBox.style.display = "block";
    } else {
      xaiBox.style.display = "none";
    }

    // Create probability chart
    createProbabilityChart(data.prediction.probabilities);

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: "smooth" });
  }

  // ============================================================
  // PROBABILITY CHART
  // ============================================================

  let probChart = null;

  function createProbabilityChart(probabilities) {
    const ctx = document.getElementById("probabilitiesChart");
    ctx.innerHTML = '<canvas id="probCanvas"></canvas>';

    const canvas = document.getElementById("probCanvas");

    if (probChart) {
      probChart.destroy();
    }

    const labels = Object.keys(probabilities);
    const values = Object.values(probabilities);

    probChart = new Chart(canvas, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Probability (%)",
            data: values,
            backgroundColor: ["#22c55e", "#f59e0b", "#ef4444"],
            borderRadius: 8,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            ticks: { color: "#94a3b8" },
            grid: { color: "rgba(255,255,255,0.1)" },
          },
          x: {
            ticks: { color: "#94a3b8" },
            grid: { display: false },
          },
        },
      },
    });
  }

  // ============================================================
  // TABS
  // ============================================================

  const tabBtns = document.querySelectorAll(".tab-btn");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      // Remove active from all
      tabBtns.forEach((b) => b.classList.remove("active"));
      document
        .querySelectorAll(".tab-pane")
        .forEach((p) => p.classList.remove("active"));

      // Add active to clicked
      btn.classList.add("active");
      const tabId = "tab-" + btn.dataset.tab;
      document.getElementById(tabId).classList.add("active");
    });
  });

  // ============================================================
  // MODEL COMPARISON
  // ============================================================

  async function loadModelComparison() {
    try {
      const response = await fetch("/model_comparison");
      const data = await response.json();

      populateComparisonTable(data.models);
      createComparisonCharts(data.models);
    } catch (error) {
      console.error("Error loading model comparison:", error);
    }
  }

  function populateComparisonTable(models) {
    const tbody = document.querySelector("#comparisonTable tbody");
    tbody.innerHTML = "";
    const deployedModelName = "HoVerNet-GNN";

    models.forEach((model) => {
      const row = document.createElement("tr");
      if (model.name === deployedModelName) {
        row.classList.add("deployed-model");
      }

      const deployedBadge =
        model.name === deployedModelName
          ? ' <span class="deployed-tag">DEPLOYED</span>'
          : "";

      row.innerHTML = `
                <td>${model.name}${deployedBadge}</td>
                <td>${model.accuracy.toFixed(2)}%</td>
                <td>${model.f1.toFixed(2)}%</td>
                <td>${model.precision.toFixed(2)}%</td>
                <td>${model.recall.toFixed(2)}%</td>
                <td>${model.auc.toFixed(3)}</td>
            `;

      tbody.appendChild(row);
    });
  }

  function createComparisonCharts(models) {
    // Accuracy Chart
    const accCtx = document.getElementById("accuracyChart");
    new Chart(accCtx, {
      type: "bar",
      data: {
        labels: models.map((m) => m.name),
        datasets: [
          {
            label: "Accuracy (%)",
            data: models.map((m) => m.accuracy),
            backgroundColor: [
              "#3498db",
              "#e74c3c",
              "#2ecc71",
              "#9b59b6",
              "#f39c12",
            ],
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            ticks: { color: "#94a3b8" },
            grid: { color: "rgba(255,255,255,0.1)" },
          },
          x: {
            ticks: { color: "#94a3b8", maxRotation: 45 },
            grid: { display: false },
          },
        },
      },
    });

    // Metrics Chart (live values, no old static/derived plot images)
    const metricsCtx = document.getElementById("metricsChart");
    new Chart(metricsCtx, {
      type: "bar",
      data: {
        labels: models.map((m) => m.name),
        datasets: [
          {
            label: "F1 Score (%)",
            data: models.map((m) => m.f1),
            backgroundColor: "#6366f1",
          },
          {
            label: "Precision (%)",
            data: models.map((m) => m.precision),
            backgroundColor: "#22c55e",
          },
          {
            label: "Recall (%)",
            data: models.map((m) => m.recall),
            backgroundColor: "#f59e0b",
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: "#94a3b8" },
          },
        },
        scales: {
          x: {
            ticks: { color: "#94a3b8", maxRotation: 45 },
            grid: { display: false },
          },
          y: {
            beginAtZero: true,
            max: 100,
            ticks: { color: "#94a3b8" },
            grid: { color: "rgba(255,255,255,0.1)" },
          },
        },
      },
    });
  }

  // Load comparison on page load
  loadModelComparison();
});
