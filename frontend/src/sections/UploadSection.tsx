import { useState } from "react";
import { motion } from "framer-motion";
import { Scan } from "lucide-react";
import { Button } from "@/components/ui/button";
import DropZone from "@/components/DropZone";
import ScannerLoader from "@/components/ScannerLoader";

const API_BASE = import.meta.env.VITE_API_URL;

export interface AnalysisResult {
  prediction: string[];
  confidence: number[];
  heatmap_path: string;
  report_path: string;
}

interface UploadSectionProps {
  onResult: (result: AnalysisResult) => void;
}

const UploadSection = ({ onResult }: UploadSectionProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!file || loading) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("image", file);  // must match backend

      const res = await fetch(`${API_BASE}/predict-xray`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Backend returned error");
      }

      const data: AnalysisResult = await res.json();
      onResult(data);

    } catch (err) {
      console.error(err);
      setError("Unable to connect to backend. Make sure Flask is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="py-24 px-6">
      <div className="max-w-xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-2xl font-bold text-foreground mb-2 text-center">
            Upload X-Ray Image
          </h2>

          <p className="text-sm text-muted-foreground mb-8 text-center">
            Upload a chest X-ray for AI-powered analysis
          </p>

          <div className="rounded-lg border border-border bg-card p-6">
            {loading ? (
              <ScannerLoader />
            ) : (
              <>
                <DropZone
                  onFileSelect={setFile}
                  selectedFile={file}
                  onClear={() => {
                    setFile(null);
                    setError(null);
                  }}
                />

                <Button
                  onClick={handleAnalyze}
                  disabled={!file}
                  className="w-full mt-4 gap-2"
                  size="lg"
                >
                  <Scan className="w-4 h-4" />
                  Analyze X-Ray
                </Button>
              </>
            )}

            {error && (
              <p className="text-xs text-red-500 mt-3 text-center">
                {error}
              </p>
            )}
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default UploadSection;
