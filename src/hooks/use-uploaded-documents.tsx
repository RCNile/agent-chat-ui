import { useState, useEffect } from "react";
import type { Base64ContentBlock } from "@langchain/core/messages";

const STORAGE_KEY = "uploaded_documents";

export interface UploadedDocument {
  id: string;
  block: Base64ContentBlock;
  uploadedAt: number;
  selected?: boolean;
}

export function useUploadedDocuments() {
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const parsedDocs = JSON.parse(stored);
        // Ensure all documents have a selected property
        const normalizedDocs = parsedDocs.map((doc: UploadedDocument) => ({
          ...doc,
          selected: doc.selected ?? false,
        }));
        setDocuments(normalizedDocs);
      } catch (error) {
        console.error("Failed to parse uploaded documents from storage:", error);
      }
    }
  }, []);

  const saveDocuments = (docs: UploadedDocument[]) => {
    setDocuments(docs);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(docs));
  };

  const addDocument = (block: Base64ContentBlock) => {
    const newDoc: UploadedDocument = {
      id: `${Date.now()}-${Math.random()}`,
      block,
      uploadedAt: Date.now(),
      selected: false,
    };
    saveDocuments([...documents, newDoc]);
  };

  const addDocuments = (blocks: Base64ContentBlock[]) => {
    const newDocs: UploadedDocument[] = blocks.map(block => ({
      id: `${Date.now()}-${Math.random()}`,
      block,
      uploadedAt: Date.now(),
      selected: false,
    }));
    saveDocuments([...documents, ...newDocs]);
  };

  const removeDocument = (id: string) => {
    saveDocuments(documents.filter(doc => doc.id !== id));
  };

  const clearAllDocuments = () => {
    saveDocuments([]);
  };

  const toggleDocumentSelection = (id: string) => {
    saveDocuments(
      documents.map(doc => 
        doc.id === id ? { ...doc, selected: !doc.selected } : doc
      )
    );
  };

  const selectAllDocuments = () => {
    saveDocuments(documents.map(doc => ({ ...doc, selected: true })));
  };

  const deselectAllDocuments = () => {
    saveDocuments(documents.map(doc => ({ ...doc, selected: false })));
  };

  const getSelectedDocuments = () => {
    return documents.filter(doc => doc.selected);
  };

  return {
    documents,
    addDocument,
    addDocuments,
    removeDocument,
    clearAllDocuments,
    toggleDocumentSelection,
    selectAllDocuments,
    deselectAllDocuments,
    getSelectedDocuments,
  };
}

