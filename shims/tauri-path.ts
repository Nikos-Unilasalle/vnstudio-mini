/** Browser stand-in for @tauri-apps/api/path. */
export const BaseDirectory = {
  Home: 1,
  AppConfig: 2,
  AppData: 3,
  Document: 4,
  Download: 5,
} as const

export async function homeDir(): Promise<string> {
  return '/home'
}

export async function join(...parts: string[]): Promise<string> {
  return parts.join('/')
}

export async function appConfigDir(): Promise<string> {
  return '/home/.vnstudio'
}
