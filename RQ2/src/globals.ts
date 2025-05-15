// globals.ts
declare let _DEVMODE: boolean

export var DEVMODE = _DEVMODE
export const ENABLE_TIMER = !_DEVMODE;  
// export const ENABLE_TIMER = true; // Set to `false` to disable the timer for debugging

export const REWARD_CORRECT = 0.1; // Reward for a correct answer
export const PENALTY_INCORRECT = -0.1; // Penalty for an incorrect answer