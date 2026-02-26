const API_BASE = "http://127.0.0.1:5000";

const ReportDownload = ({ reportPath }) => {
  if (!reportPath) return null;

  const fileUrl = `${API_BASE}/${reportPath}`;

  return (
    <a
      href={fileUrl}
      target="_blank"
      rel="noopener noreferrer"
      className="download-btn"
    >
      Download PDF Report
    </a>
  );
};

export default ReportDownload;
