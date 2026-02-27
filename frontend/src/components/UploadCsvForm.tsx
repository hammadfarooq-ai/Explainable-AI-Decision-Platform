import React, { useState } from "react";
import { api } from "../api/client";

interface Props {
  onUploaded: (datasetId: number, summary: any) => void;
}

export const UploadCsvForm: React.FC<Props> = ({ onUploaded }) => {
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !target) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const uploadRes = await api.post("/ml/datasets/upload", formData, {
        params: { target_column: target },
        headers: { "Content-Type": "multipart/form-data" },
      });

      const datasetId = uploadRes.data.dataset_id;
      const edaRes = await api.get(`/ml/datasets/${datasetId}/eda`);
      onUploaded(datasetId, edaRes.data);
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <input
          type="file"
          accept=".csv"
          onChange={(e) => {
            const f = e.target.files?.[0] ?? null;
            setFile(f);
          }}
        />
      </div>
      <div>
        <input
          type="text"
          placeholder="Target column"
          value={target}
          onChange={(e) => setTarget(e.target.value)}
        />
      </div>
      <button type="submit" disabled={loading}>
        {loading ? "Uploading..." : "Upload"}
      </button>
      {error && <p style={{ color: "red" }}>{error}</p>}
    </form>
  );
};

