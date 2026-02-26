import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle, Download } from "lucide-react";
import ConfidenceBar from "./ConfidenceBar";

interface PredictionCardProps {
  predictions: string[];
  confidences: number[];
  reportPath?: string;
}

const PredictionCard = ({
  predictions,
  confidences,
  reportPath,
}: PredictionCardProps) => {
  const isNormal = predictions.length === 1 && predictions[0] === "Normal";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="rounded-lg border border-border bg-card p-6 clinical-glow"
    >
      <div className="flex items-center gap-3 mb-6">
        {isNormal ? (
          <CheckCircle className="w-5 h-5 text-success" />
        ) : (
          <AlertTriangle className="w-5 h-5 text-warning" />
        )}
        <h3 className="text-lg font-semibold text-foreground">
          {isNormal ? "No Abnormalities Detected" : "Findings Detected"}
        </h3>
      </div>

      {isNormal ? (
        <div className="rounded-md bg-success/10 border border-success/20 p-4">
          <p className="text-sm text-foreground">
            The analysis indicates a{" "}
            <span className="font-semibold text-success">normal</span> chest
            X-ray with{" "}
            <span className="font-mono">
              {Math.round(confidences[0] * 100)}%
            </span>{" "}
            confidence.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            {predictions.length} condition
            {predictions.length > 1 ? "s" : ""} identified
          </p>

          {predictions.map((pred, i) => (
            <ConfidenceBar
              key={pred}
              label={pred}
              value={confidences[i]}
              index={i}
            />
          ))}
        </div>
      )}

      {/* 🔥 Download Button Section */}
      {reportPath && (
        <div className="mt-6 flex justify-center">
          <a
            href={`http://127.0.0.1:5000/${reportPath}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground hover:opacity-90 transition-all"
          >
            <Download className="w-4 h-4" />
            Download PDF Report
          </a>
        </div>
      )}
    </motion.div>
  );
};

export default PredictionCard;
