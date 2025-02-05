/* tslint:disable */
/* eslint-disable */
export class Machine {
  free(): void;
  constructor(input: string);
  space_time(goal_x: number, goal_y: number, early_stop?: number | null): SpaceTimeResult;
  count_ones(): number;
  is_halted(): boolean;
  count_steps(early_stop?: number | null): number;
}
export class SpaceTimeResult {
  private constructor();
  free(): void;
  png_data(): Uint8Array;
  step_count(): number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_machine_free: (a: number, b: number) => void;
  readonly machine_from_string: (a: number, b: number) => [number, number, number];
  readonly machine_space_time: (a: number, b: number, c: number, d: number) => number;
  readonly machine_count_ones: (a: number) => number;
  readonly machine_is_halted: (a: number) => number;
  readonly machine_count_steps: (a: number, b: number) => number;
  readonly __wbg_spacetimeresult_free: (a: number, b: number) => void;
  readonly spacetimeresult_png_data: (a: number) => [number, number];
  readonly spacetimeresult_step_count: (a: number) => number;
  readonly __wbindgen_export_0: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
