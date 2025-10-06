import { Button } from "@/components/ui/button";
import { useThreads } from "@/providers/Thread";
import { Thread } from "@langchain/langgraph-sdk";
import { useEffect, useState } from "react";
import type { MouseEvent } from "react";

import { getContentString } from "../utils";
import { useQueryState, parseAsBoolean } from "nuqs";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { PanelRightOpen, PanelRightClose, Trash2 } from "lucide-react";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { createClient } from "@/providers/client";
import { getApiKey } from "@/lib/api-key";
import { toast } from "sonner";

function ThreadList({
  threads,
  onThreadClick,
  onThreadDelete,
}: {
  threads: Thread[];
  onThreadClick?: (threadId: string) => void;
  onThreadDelete?: (threadId: string) => void;
}) {
  const [threadId, setThreadId] = useQueryState("threadId");
  const [apiUrl] = useQueryState("apiUrl");
  const [assistantId] = useQueryState("assistantId");
  const [deletingThreadId, setDeletingThreadId] = useState<string | null>(null);

  const handleThreadClick = (newThreadId: string) => {
    onThreadClick?.(newThreadId);
    if (newThreadId === threadId) return;
    
    // Preserve apiUrl and assistantId in URL when switching threads
    const params = new URLSearchParams(window.location.search);
    if (apiUrl) params.set('apiUrl', apiUrl);
    if (assistantId) params.set('assistantId', assistantId);
    params.set('threadId', newThreadId);
    
    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, '', newUrl);
    
    setThreadId(newThreadId);
  };

  const handleDeleteThread = async (threadIdToDelete: string, e: MouseEvent) => {
    e.stopPropagation(); // Prevent thread selection
    
    if (!window.confirm("Are you sure you want to delete this thread? This action cannot be undone.")) {
      return;
    }

    setDeletingThreadId(threadIdToDelete);
    
    try {
      if (!apiUrl) {
        toast.error("Error", {
          description: "API URL not configured",
          duration: 3000,
        });
        return;
      }

      const client = createClient(apiUrl, getApiKey() ?? undefined);
      await client.threads.delete(threadIdToDelete);
      
      toast.success("Thread deleted successfully");
      onThreadDelete?.(threadIdToDelete);
      
      // If we deleted the current thread, clear it from URL
      if (threadIdToDelete === threadId) {
        const params = new URLSearchParams(window.location.search);
        if (apiUrl) params.set('apiUrl', apiUrl);
        if (assistantId) params.set('assistantId', assistantId);
        params.delete('threadId');
        
        const newUrl = `${window.location.pathname}?${params.toString()}`;
        window.history.replaceState({}, '', newUrl);
        setThreadId(null);
      }
    } catch (error) {
      console.error("Failed to delete thread:", error);
      toast.error("Failed to delete thread", {
        description: "Please try again",
        duration: 3000,
      });
    } finally {
      setDeletingThreadId(null);
    }
  };

  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {threads.map((t) => {
        let itemText = t.thread_id;
        if (
          typeof t.values === "object" &&
          t.values &&
          "messages" in t.values &&
          Array.isArray(t.values.messages) &&
          t.values.messages?.length > 0
        ) {
          const firstMessage = t.values.messages[0];
          itemText = getContentString(firstMessage.content);
        }
        const isDeleting = deletingThreadId === t.thread_id;
        const isCurrentThread = t.thread_id === threadId;
        
        return (
          <div
            key={t.thread_id}
            className="group relative w-full px-1"
          >
            <Button
              variant="ghost"
              className={`w-[280px] items-start justify-start text-left font-normal ${
                isCurrentThread ? "bg-accent" : ""
              }`}
              onClick={(e) => {
                e.preventDefault();
                handleThreadClick(t.thread_id);
              }}
            >
              <p className="truncate text-ellipsis pr-8">{itemText}</p>
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="absolute right-1 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity p-1 h-6 w-6 text-muted-foreground hover:text-destructive"
              onClick={(e) => handleDeleteThread(t.thread_id, e)}
              disabled={isDeleting}
            >
              {isDeleting ? (
                <div className="h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
              ) : (
                <Trash2 className="h-3 w-3" />
              )}
            </Button>
          </div>
        );
      })}
    </div>
  );
}

function ThreadHistoryLoading() {
  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {Array.from({ length: 30 }).map((_, i) => (
        <Skeleton
          key={`skeleton-${i}`}
          className="h-10 w-[280px]"
        />
      ))}
    </div>
  );
}

export default function ThreadHistory() {
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );

  const { getThreads, threads, setThreads, threadsLoading, setThreadsLoading } =
    useThreads();

  useEffect(() => {
    if (typeof window === "undefined") return;
    setThreadsLoading(true);
    getThreads()
      .then(setThreads)
      .catch(console.error)
      .finally(() => setThreadsLoading(false));
  }, [getThreads]);

  const handleThreadDelete = (deletedThreadId: string) => {
    // Remove the deleted thread from the local state
    setThreads(prevThreads => prevThreads.filter(t => t.thread_id !== deletedThreadId));
  };

  return (
    <>
      <div className="shadow-inner-right hidden h-screen w-[300px] shrink-0 flex-col items-start justify-start gap-6 border-r-[1px] border-slate-300 lg:flex">
        <div className="flex w-full items-center justify-between px-4 pt-1.5">
          <Button
            className="hover:bg-gray-100"
            variant="ghost"
            onClick={() => setChatHistoryOpen((p) => !p)}
          >
            {chatHistoryOpen ? (
              <PanelRightOpen className="size-5" />
            ) : (
              <PanelRightClose className="size-5" />
            )}
          </Button>
          <h1 className="text-xl font-semibold tracking-tight">
            Thread History
          </h1>
        </div>
        {threadsLoading ? (
          <ThreadHistoryLoading />
        ) : (
          <ThreadList 
            threads={threads} 
            onThreadDelete={handleThreadDelete}
          />
        )}
      </div>
      <div className="lg:hidden">
        <Sheet
          open={!!chatHistoryOpen && !isLargeScreen}
          onOpenChange={(open) => {
            if (isLargeScreen) return;
            setChatHistoryOpen(open);
          }}
        >
          <SheetContent
            side="left"
            className="flex lg:hidden"
          >
            <SheetHeader>
              <SheetTitle>Thread History</SheetTitle>
            </SheetHeader>
            <ThreadList
              threads={threads}
              onThreadClick={() => setChatHistoryOpen((o) => !o)}
              onThreadDelete={handleThreadDelete}
            />
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
