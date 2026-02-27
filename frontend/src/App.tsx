import React, { useState } from "react";
import { UploadCsvForm } from "./components/UploadCsvForm";
import { EdaSummary } from "./components/EdaSummary";
import { RagQueryForm } from "./components/RagQueryForm";
import { PredictForm } from "./components/PredictForm";

export const App: React.FC = () => {
  const [datasetId, setDatasetId] = useState<number | null>(null);
  const [edaSummary, setEdaSummary] = useState<any | null>(null);

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "1rem" }}>
      <h1>Enterprise AI Decision Platform</h1>

      <section>
        <h2>1. Upload Dataset</h2>
        <UploadCsvForm
          onUploaded={(id, summary) => {
            setDatasetId(id);
            setEdaSummary(summary);
          }}
        />
      </section>

      <section>
        <h2>2. EDA Summary</h2>
        <EdaSummary summary={edaSummary} />
      </section>

      <section>
        <h2>3. Predict</h2>
        <PredictForm />
      </section>

      <section>
        <h2>4. RAG Recommendations</h2>
        <RagQueryForm />
      </section>
    </div>
  );
};

