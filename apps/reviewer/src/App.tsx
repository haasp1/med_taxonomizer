import React, { ChangeEvent, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Download, FileJson, Search, Upload } from "lucide-react";
import "./style.css";

type ReviewDecision = "" | "approve" | "rename" | "merge" | "split" | "reject";

type ReviewNode = {
  id: string;
  label: string;
  definition: string;
  path: string;
  parent: string;
  support: string;
  decision: ReviewDecision;
  newLabel: string;
  newDefinition: string;
  mergeInto: string;
  splitNotes: string;
  reviewerNotes: string;
};

const sampleNodes: ReviewNode[] = [
  {
    id: "node-001",
    label: "Bleeding complication",
    definition: "Bleeding or hemorrhage described after a procedure or treatment.",
    path: "complications/bleeding",
    parent: "complications",
    support: "12",
    decision: "",
    newLabel: "",
    newDefinition: "",
    mergeInto: "",
    splitNotes: "",
    reviewerNotes: "",
  },
  {
    id: "node-002",
    label: "Infectious complication",
    definition: "Fever, suspected infection, or infection treatment mentioned in the text.",
    path: "complications/infection",
    parent: "complications",
    support: "8",
    decision: "",
    newLabel: "",
    newDefinition: "",
    mergeInto: "",
    splitNotes: "",
    reviewerNotes: "",
  },
];

function stringify(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}

function pick(record: Record<string, unknown>, keys: string[]): string {
  for (const key of keys) {
    if (record[key] !== undefined && record[key] !== null && record[key] !== "") return stringify(record[key]);
  }
  return "";
}

function collectNodes(value: unknown, out: Record<string, unknown>[] = []): Record<string, unknown>[] {
  if (Array.isArray(value)) {
    for (const item of value) collectNodes(item, out);
    return out;
  }
  if (!value || typeof value !== "object") return out;
  const record = value as Record<string, unknown>;
  const hasLabel = ["label", "name", "title", "category", "code"].some((key) => key in record);
  const hasTaxonomyShape = ["node_id", "id", "path", "definition", "description", "children"].some((key) => key in record);
  if (hasLabel && hasTaxonomyShape) out.push(record);
  for (const key of ["nodes", "children", "categories", "subcategories", "leaves", "taxonomy"]) {
    if (key in record) collectNodes(record[key], out);
  }
  return out;
}

function normalizeNodes(json: unknown): ReviewNode[] {
  const records = collectNodes(json);
  return records.map((record, index) => {
    const label = pick(record, ["label", "name", "title", "category", "code"]);
    const path = pick(record, ["path", "taxonomy_path", "full_path", "code"]);
    return {
      id: pick(record, ["node_id", "id", "stable_id", "code"]) || `node-${String(index + 1).padStart(3, "0")}`,
      label: label || path || `Node ${index + 1}`,
      definition: pick(record, ["definition", "description", "rationale"]),
      path,
      parent: pick(record, ["parent", "parent_id", "parent_path", "meta"]),
      support: pick(record, ["support", "provenance_count", "count", "n"]),
      decision: "",
      newLabel: "",
      newDefinition: "",
      mergeInto: "",
      splitNotes: "",
      reviewerNotes: "",
    };
  });
}

function csvEscape(value: string): string {
  const normalized = value.replace(/\r?\n/g, " ");
  if (/[",\n]/.test(normalized)) return `"${normalized.replace(/"/g, '""')}"`;
  return normalized;
}

function exportCsv(nodes: ReviewNode[]): string {
  const header = [
    "node_id",
    "current_label",
    "current_definition",
    "path",
    "parent",
    "support",
    "decision",
    "new_label",
    "new_definition",
    "merge_into_node_id",
    "split_notes",
    "reviewer_notes",
  ];
  const rows = nodes.map((node) => [
    node.id,
    node.label,
    node.definition,
    node.path,
    node.parent,
    node.support,
    node.decision,
    node.newLabel,
    node.newDefinition,
    node.mergeInto,
    node.splitNotes,
    node.reviewerNotes,
  ]);
  return [header, ...rows].map((row) => row.map(csvEscape).join(",")).join("\n") + "\n";
}

function downloadFile(filename: string, content: string, type: string): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function App() {
  const [nodes, setNodes] = useState<ReviewNode[]>(sampleNodes);
  const [query, setQuery] = useState("");
  const [message, setMessage] = useState("Loaded synthetic example nodes. Upload a taxonomy JSON file to review your own tree locally.");

  const filteredNodes = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return nodes;
    return nodes.filter((node) => [node.id, node.label, node.definition, node.path, node.parent].join(" ").toLowerCase().includes(q));
  }, [nodes, query]);

  const reviewedCount = nodes.filter((node) => node.decision).length;

  async function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      const normalized = normalizeNodes(parsed);
      if (!normalized.length) {
        setMessage("No taxonomy-like nodes were found. Expected labels, paths, definitions, or nested children.");
        return;
      }
      setNodes(normalized);
      setMessage(`Loaded ${normalized.length} nodes from ${file.name}. Everything stays in this browser session.`);
    } catch (error) {
      setMessage(error instanceof Error ? `Could not read JSON: ${error.message}` : "Could not read JSON file.");
    }
  }

  function updateNode(id: string, patch: Partial<ReviewNode>) {
    setNodes((current) => current.map((node) => (node.id === id ? { ...node, ...patch } : node)));
  }

  return (
    <main>
      <section className="hero">
        <p className="eyebrow">Local taxonomy review</p>
        <h1>Med Taxonomizer Reviewer</h1>
        <p className="lead">
          A small local browser app for checking a taxonomy tree before full-corpus labeling. It runs on your machine and
          does not upload taxonomy files or review notes.
        </p>
        <div className="actions">
          <label className="button">
            <Upload size={18} />
            Upload taxonomy JSON
            <input type="file" accept=".json,application/json" onChange={onFileChange} />
          </label>
          <button className="secondary" onClick={() => downloadFile("taxonomy_review_queue.csv", exportCsv(nodes), "text/csv")}> 
            <Download size={18} />
            Export review CSV
          </button>
        </div>
      </section>

      <section className="status" aria-live="polite">
        <FileJson />
        <span>{message}</span>
      </section>

      <section className="toolbar">
        <div>
          <strong>{nodes.length}</strong> nodes loaded
        </div>
        <div>
          <strong>{reviewedCount}</strong> reviewed
        </div>
        <label className="search">
          <Search size={16} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search label, path, or definition" />
        </label>
      </section>

      <section className="review-list">
        {filteredNodes.map((node) => (
          <article className="node-card" key={node.id}>
            <div className="node-head">
              <div>
                <p className="node-id">{node.id}</p>
                <h2>{node.label}</h2>
                {node.path ? <p className="path">{node.path}</p> : null}
              </div>
              <select value={node.decision} onChange={(event) => updateNode(node.id, { decision: event.target.value as ReviewDecision })}>
                <option value="">Decision</option>
                <option value="approve">Approve</option>
                <option value="rename">Rename</option>
                <option value="merge">Merge</option>
                <option value="split">Split</option>
                <option value="reject">Reject</option>
              </select>
            </div>
            <p className="definition">{node.definition || "No definition provided."}</p>
            <div className="meta-grid">
              <span>Parent: {node.parent || "none"}</span>
              <span>Support: {node.support || "not provided"}</span>
            </div>
            <div className="form-grid">
              <input value={node.newLabel} onChange={(event) => updateNode(node.id, { newLabel: event.target.value })} placeholder="New label if renamed" />
              <input value={node.mergeInto} onChange={(event) => updateNode(node.id, { mergeInto: event.target.value })} placeholder="Merge into node ID" />
              <textarea value={node.newDefinition} onChange={(event) => updateNode(node.id, { newDefinition: event.target.value })} placeholder="New or clarified definition" />
              <textarea value={node.splitNotes} onChange={(event) => updateNode(node.id, { splitNotes: event.target.value })} placeholder="Split notes" />
              <textarea className="wide" value={node.reviewerNotes} onChange={(event) => updateNode(node.id, { reviewerNotes: event.target.value })} placeholder="Reviewer notes" />
            </div>
          </article>
        ))}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
