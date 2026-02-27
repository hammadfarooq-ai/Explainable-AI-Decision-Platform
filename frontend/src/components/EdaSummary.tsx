import React from "react";

interface Props {
  summary: any | null;
}

export const EdaSummary: React.FC<Props> = ({ summary }) => {
  if (!summary) {
    return <p>No EDA summary yet.</p>;
  }

  return (
    <div style={{ fontSize: "0.9rem" }}>
      <pre>{JSON.stringify(summary, null, 2)}</pre>
    </div>
  );
};

