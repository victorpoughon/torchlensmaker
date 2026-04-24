// One scene received from tlmserver or a workspace file, as stored in the scene list.
export interface SceneEntry {
  id: string
  topic: string
  timestamp: Date
  payload: unknown
}
