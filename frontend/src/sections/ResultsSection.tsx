import { motion } from "framer-motion";
import PredictionCard from "@/components/PredictionCard";
import HeatmapViewer from "@/components/HeatmapViewer";
import type { AnalysisResult } from "./UploadSection";

interface ResultsSectionProps {
  result: AnalysisResult | null;
}

const ResultsSection = ({ result }: ResultsSectionProps) => {
  if (!result) return null;

  return (
    <section className="py-24 px-6">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-2xl font-bold text-foreground mb-2 text-center">
            Analysis Results
          </h2>
          <p className="text-sm text-muted-foreground mb-10 text-center">
            AI-generated diagnostic findings
          </p>

          <div className="grid md:grid-cols-2 gap-6">
            <PredictionCard
              predictions={result.prediction}
              confidences={result.confidence}
              reportPath={result.report_path}
            />
            <HeatmapViewer heatmapPath={result.heatmap_path} />
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default ResultsSection;
