// One scene received from tlmserver or a workspace file, as stored in the scene list.
export interface SceneEntry {
  id: string
  topic: string
  timestamp: Date
  payload: unknown
}

// One source file received from tlmserver, displayed in a Source panel.
export interface SourceEntry {
  id: string
  filename: string
  language: string
  content: string
  timestamp: Date
}
