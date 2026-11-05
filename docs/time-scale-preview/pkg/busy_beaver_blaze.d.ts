/* tslint:disable */
/* eslint-disable */
export class Machine {
  free(): void;
  /**
   * Creates a new Machine from a program string.
   *
   * # Errors
   *
   * Returns an error if the program string cannot be parsed into a valid Turing machine program.
   */
  constructor(program: string);
  step(): boolean;
  /**
   * # Panics
   * Panics if the number of nonblank cells exceeds `u32::MAX`.
   */
  count_nonblanks(): number;
  is_halted(): boolean;
  count(early_stop_is_some: boolean, early_stop_number: bigint): bigint;
  state(): number;
  tape_index(): bigint;
}
/**
 * Separate experimental stepping path: the reference machine has no history checks.
 */
export class PyramidMachine {
  free(): void;
  /**
   * Start an experimental Quality run with native multilayer history capture.
   */
  constructor(program: string, x_goal: number, y_goal: number, scale: string);
  /**
   * Change reconstruction without rerunning the simulation.
   */
  set_time_scale(scale: string): void;
  /**
   * Change the horizontal coordinate map without stepping or capturing history.
   */
  set_space_scale(scale: string): void;
  /**
   * Step in bounded batches, capturing exact checkpoints even inside a batch.
   */
  step_for_secs(seconds: number, early_stop: bigint | null | undefined, loops_per_time_check: bigint): boolean;
  /**
   * Advance one transition using the same halt convention as Linear A.
   */
  step(): boolean;
  /**
   * Encode reconstructed symbol layers using the existing PNG encoder.
   */
  to_png(colors: Uint8Array): Uint8Array;
  /**
   * Render the active reference image while keeping pyramid history available.
   */
  linear_png(colors: Uint8Array): Uint8Array;
  /**
   * Current sample count, matching the reference API (initial row included).
   */
  step_count(): bigint;
  /**
   * Number of nonblank tape cells.
   */
  count_nonblanks(): number;
  /**
   * Whether the underlying program halted.
   */
  is_halted(): boolean;
  /**
   * Number of cumulative checkpoints, including the initial blank image.
   */
  checkpoint_count(): number;
  /**
   * Estimated retained layer bytes, including allocation capacity and row structures.
   * Excludes allocator metadata, tape, PNG buffers and transient reconstruction storage.
   */
  retained_bytes(): number;
}
export class SpaceByTimeMachine {
  free(): void;
  /**
   * Creates a `SpaceByTimeMachine` by parsing the program, optionally skipping initial steps, and initializing visualization layers for the given goals and pixel policy.
   *
   * # Errors
   * Returns an error if the program cannot be parsed into a `Machine`, or if the machine halts during the initial skip phase.
   */
  constructor(program: string, goal_x: number, goal_y: number, binning: boolean, skip: bigint);
  nth(n: bigint): boolean;
  step_for_secs(seconds: number, early_stop: bigint | null | undefined, loops_per_time_check: bigint): boolean;
  /**
   * Encodes the current space-time visualization into a PNG byte vector using the provided RGB palette.
   *
   * The `colors` slice must have a length that is a multiple of 3 (RGB triplets); image dimensions are taken from the first layer.
   *
   * # Errors
   * Returns an error if the `colors` length is not a multiple of 3 or if PNG packing fails in the underlying layers.
   */
  to_png(colors: Uint8Array): Uint8Array;
  /**
   * Apply an independent tape-coordinate map to the reference symbol layers.
   * Linear space uses the original PNG fast path; nonlinear space needs no pyramid.
   */
  to_spatial_png(colors: Uint8Array, space_scale: string): Uint8Array;
  step_count(): bigint;
  count_nonblanks(): number;
  is_halted(): boolean;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_spacebytimemachine_free: (a: number, b: number) => void;
  readonly spacebytimemachine_from_str: (a: number, b: number, c: number, d: number, e: number, f: bigint) => [number, number, number];
  readonly spacebytimemachine_nth: (a: number, b: bigint) => number;
  readonly spacebytimemachine_step_for_secs: (a: number, b: number, c: number, d: bigint, e: bigint) => number;
  readonly spacebytimemachine_to_png: (a: number, b: number, c: number) => [number, number, number, number];
  readonly spacebytimemachine_to_spatial_png: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
  readonly spacebytimemachine_step_count: (a: number) => bigint;
  readonly spacebytimemachine_count_nonblanks: (a: number) => number;
  readonly spacebytimemachine_is_halted: (a: number) => number;
  readonly __wbg_pyramidmachine_free: (a: number, b: number) => void;
  readonly pyramidmachine_new: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
  readonly pyramidmachine_set_time_scale: (a: number, b: number, c: number) => [number, number];
  readonly pyramidmachine_set_space_scale: (a: number, b: number, c: number) => [number, number];
  readonly pyramidmachine_step_for_secs: (a: number, b: number, c: number, d: bigint, e: bigint) => number;
  readonly pyramidmachine_step: (a: number) => number;
  readonly pyramidmachine_to_png: (a: number, b: number, c: number) => [number, number, number, number];
  readonly pyramidmachine_linear_png: (a: number, b: number, c: number) => [number, number, number, number];
  readonly pyramidmachine_step_count: (a: number) => bigint;
  readonly pyramidmachine_count_nonblanks: (a: number) => number;
  readonly pyramidmachine_is_halted: (a: number) => number;
  readonly pyramidmachine_checkpoint_count: (a: number) => number;
  readonly pyramidmachine_retained_bytes: (a: number) => number;
  readonly __wbg_machine_free: (a: number, b: number) => void;
  readonly machine_from_string: (a: number, b: number) => [number, number, number];
  readonly machine_step: (a: number) => number;
  readonly machine_count_nonblanks: (a: number) => number;
  readonly machine_is_halted: (a: number) => number;
  readonly machine_count: (a: number, b: number, c: bigint) => bigint;
  readonly machine_state: (a: number) => number;
  readonly machine_tape_index: (a: number) => bigint;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
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
