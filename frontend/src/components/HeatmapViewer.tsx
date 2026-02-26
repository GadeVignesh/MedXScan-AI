const API_BASE = "http://127.0.0.1:5000";

const HeatmapViewer = ({ heatmapPath }) => {
  if (!heatmapPath) return null;

  const imageUrl = `${API_BASE}/${heatmapPath}`;

  return (
    <div className="heatmap-card">
      <h3>Grad-CAM Heatmap</h3>
      <img
        src={imageUrl}
        alt="GradCAM"
        style={{ width: "100%", borderRadius: "12px" }}
      />
    </div>
  );
};

export default HeatmapViewer;
