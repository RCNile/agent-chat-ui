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
    // Filter out blocks that are already in the library
    const newBlocks = blocks.filter(block => {
      const filename = block.metadata?.filename || block.metadata?.name;
      const isDuplicate = documents.some(doc => {
        const existingFilename = doc.block.metadata?.filename || doc.block.metadata?.name;
        return (
          existingFilename === filename &&
          doc.block.mime_type === block.mime_type &&
          doc.block.data === block.data
        );
      });
      return !isDuplicate;
    });

    if (newBlocks.length === 0) return;

    const newDocs: UploadedDocument[] = newBlocks.map(block => ({
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
    const updatedDocs = documents.map(doc => 
      doc.id === id ? { ...doc, selected: !doc.selected } : doc
    );
    console.log("Toggling document selection:", {
      id,
      documentName: documents.find(d => d.id === id)?.block.metadata?.filename,
      newState: updatedDocs.find(d => d.id === id)?.selected,
      allDocumentsAfterToggle: updatedDocs.map(d => ({
        filename: d.block.metadata?.filename || d.block.metadata?.name,
        selected: d.selected
      }))
    });
    saveDocuments(updatedDocs);
  };

  const selectAllDocuments = () => {
    saveDocuments(documents.map(doc => ({ ...doc, selected: true })));
  };

  const deselectAllDocuments = () => {
    saveDocuments(documents.map(doc => ({ ...doc, selected: false })));
  };

  const getSelectedDocuments = () => {
    // Read directly from localStorage to get the most up-to-date state
    // This avoids issues with React state update timing
    const stored = localStorage.getItem(STORAGE_KEY);
    let currentDocs = documents;
    
    if (stored) {
      try {
        currentDocs = JSON.parse(stored);
      } catch (error) {
        console.error("Failed to parse documents from storage:", error);
      }
    }
    
    const selected = currentDocs.filter(doc => doc.selected);
    console.log("All documents (from localStorage):", currentDocs.map(d => ({
      filename: d.block.metadata?.filename || d.block.metadata?.name,
      selected: d.selected
    })));
    console.log("Filtered selected documents:", selected.map(d => ({
      filename: d.block.metadata?.filename || d.block.metadata?.name,
      selected: d.selected
    })));
    return selected;
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

