import { CircleAlert } from 'lucide-react'

export function Status({ children, tone = 'muted', dot = false }) {
  return <span className={`badge ${tone}`}>{dot && <span className="status-dot" />}{children}</span>
}

export function PanelTitle({ icon: Icon, title, children, description }) {
  return <div className="panel-heading">
    <div className="panel-title"><Icon size={19} /><div><h2>{title}</h2>{description && <p>{description}</p>}</div></div>
    {children}
  </div>
}

export function ErrorNote({ children }) {
  return <p className="notice error" role="alert"><CircleAlert size={17} />{children}</p>
}
