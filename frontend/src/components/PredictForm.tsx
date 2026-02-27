import React, { useState } from "react";
import { api } from "../api/client";

export const PredictForm: React.FC = () => {
  const [payload, setPayload] = useState('[{"feature_num_1": 1.0, "feature_num_2": 10.5, "feature_cat": "A"}]');
  const [result, setResult] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const records = JSON.parse(payload);
      const res = await api.post("/ml/predict", { records, explain: true });
      setResult(res.data);
    } catch (err: any) {
      setError(err?.message ?? "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <textarea
          rows={4}
          cols={60}
          value={payload}
          onChange={(e) => setPayload(e.target.value)}
        />
        <br />
        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>
      {error && <p style={{ color: "red" }}>{error}</p>}
      {result && (
        <pre style={{ fontSize: "0.9rem" }}>{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  );
};

