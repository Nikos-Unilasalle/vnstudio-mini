/**
 * One fetch wrapper for every remote source, so a failure names its cause.
 *
 * A cross-origin fetch that never reaches the server rejects with a bare
 * `TypeError` — "NetworkError when attempting to fetch resource" in Firefox,
 * "Failed to fetch" in Chrome — carrying no hint of which host was involved or
 * why. Surfaced through a node that talks to four different services, that
 * message is close to useless. This attaches the host and the short list of
 * things that actually produce it.
 */

/**
 * The services these nodes talk to.
 *
 * Each carries the URL the reachability sweep should ask for. It has to be a
 * path the host actually serves, not the bare domain: several of these answer
 * nothing at `/`, which would make the sweep report a healthy service as
 * blocked — a diagnostic that lies is worse than none at all.
 */
const KNOWN_HOSTS: Record<string, { service: string; probe: string }> = {
  'identity.dataspace.copernicus.eu': {
    service: 'authentification Copernicus CDSE',
    probe: 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/.well-known/openid-configuration',
  },
  'sh.dataspace.copernicus.eu': {
    service: 'Sentinel Hub CDSE',
    probe: 'https://sh.dataspace.copernicus.eu/api/v1/process',
  },
  'planetarycomputer.microsoft.com': {
    service: 'Microsoft Planetary Computer',
    probe: 'https://planetarycomputer.microsoft.com/api/stac/v1',
  },
  'mt1.google.com': {
    service: 'tuiles Google',
    probe: 'https://mt1.google.com/vt/lyrs=s&x=0&y=0&z=0',
  },
  'tile.openstreetmap.org': {
    service: 'tuiles OpenStreetMap',
    probe: 'https://tile.openstreetmap.org/0/0/0.png',
  },
  'basemaps.cartocdn.com': {
    service: 'tuiles Carto',
    probe: 'https://basemaps.cartocdn.com/light_all/0/0/0.png',
  },
}

function hostOf(url: string): string {
  try {
    return new URL(url).host
  } catch {
    return url
  }
}

export class RemoteError extends Error {
  constructor(message: string, readonly host: string, readonly unreachable: boolean) {
    super(message)
    this.name = 'RemoteError'
  }
}

/**
 * `fetch`, but a network-level rejection becomes a message that says which
 * service could not be reached and what usually causes that.
 */
export async function request(url: string, init?: RequestInit): Promise<Response> {
  const host = hostOf(url)
  try {
    return await fetch(url, init)
  } catch (error) {
    const service = KNOWN_HOSTS[host]?.service ?? host
    throw new RemoteError(
      `${service} injoignable (${host}) — le navigateur n’a reçu aucune réponse. ` +
        'Causes habituelles : un bloqueur de contenu ou la protection renforcée de Firefox, ' +
        'un proxy d’établissement, ou une connexion coupée. ' +
        `Détail du navigateur : ${error instanceof Error ? error.message : String(error)}`,
      host,
      true
    )
  }
}

/**
 * Which of the services these nodes use are reachable right now.
 *
 * Run only after something has already failed. When one host is blocked and the
 * others answer, the cause is almost always a blocklist entry or a proxy rule
 * rather than a broken connection — and knowing which host it is turns an
 * unactionable "NetworkError" into something the user can go and allow.
 */
export async function describeReachability(): Promise<string> {
  const probes = Object.values(KNOWN_HOSTS).map(async ({ service, probe }) => {
    try {
      // `no-cors` keeps this a reachability test rather than a CORS test: an
      // opaque response still proves the request left the browser and came back.
      await fetch(probe, { mode: 'no-cors', signal: AbortSignal.timeout(5000) })
      return `${service} : joignable`
    } catch {
      return `${service} : BLOQUÉ`
    }
  })
  const lines = await Promise.all(probes)
  return lines.join(' · ')
}
