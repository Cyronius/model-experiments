export interface Titler {
  /** A short title for the prompt. `beams`: 4 is the default, 1 is fastest, the maximum is 8. */
  title(prompt: string, options?: { beams?: number }): string;
  /** Release the model's memory. The object cannot be used afterwards. */
  free(): void;
}

export interface CreateOptions {
  /** URL, file path (Node), or GGUF bytes. Default: the model file in this package. */
  model?: string | URL | ArrayBuffer | Uint8Array;
}

export function createTitler(options?: CreateOptions): Promise<Titler>;
