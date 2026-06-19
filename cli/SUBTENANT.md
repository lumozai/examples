# Subtenant CLI

A small `bash` wrapper around the `/admin/subtenants` API that lets a **tenant org admin** create and manage subtenants from the command line.

A *subtenant* is a customer-of-a-customer: an isolated organization nested under your tenant. Subtenants are always **corporate** orgs.

Provisioning is **two independent steps**:

1. **Create the subtenant org** (`create`) — provisions the org only. No users.
2. **Add users** (`add-user`) — invite/provision users into an existing subtenant org, one at a time.

The CLI talks to the Lumoz production API (`https://api.lumoz.ai`) out of the box — no configuration needed (override with `LUMOZ_API_URL`).

## Prerequisites

- `bash`, `curl`, and `python3`.
- `jq` is optional — if present, responses are pretty-printed.
- Credentials — either an M2M key or an admin JWT (see below).

## Authentication

The CLI supports the API's two auth modes. **M2M is preferred when set**; otherwise it falls back to a JWT.

### Option 1 — M2M key (recommended for automation)

Set `LUMOZ_API_KEY` to your machine-to-machine credential as `"<client_id>:<client_secret>"` (both halves joined by a single `:`). The CLI base64-encodes it as HTTP Basic auth.

```bash
export LUMOZ_API_KEY="m2m-client-...:secret-..."
./subtenant list
```

The M2M client must carry the required scopes:
- `write:subtenants` — for `create`, `add-user`, `del-user`, `disable`, `enable`, `delete`.
- `read:subtenants` (or `write:subtenants`) — for `list`, `list-users`.

### Option 2 — admin JWT (`.auth-token`)

If `LUMOZ_API_KEY` is not set, the CLI reads a JWT from a file named `.auth-token` next to the `subtenant` script. This is your live console session token, copied from the browser — you must be signed in as a **tenant org admin**.

1. Log in to the console at <https://console.lumoz.ai>.
2. Open **DevTools → Network**, then click around or reload so a request appears.
3. Click any API request, find the **`Authorization`** request header (`Bearer eyJ...`), and copy its value **without the `Bearer ` prefix**.
4. Save just the `eyJ...` token to `.auth-token`:

   ```bash
   echo 'eyJhbGciOi...your.jwt.here' > .auth-token
   ```

5. Verify with `./subtenant list` — a `<<< HTTP 200` means you're in. A `401`/`403` means the token is missing, expired, or not an admin token; grab a fresh one.

> JWTs expire — when calls start returning `401`, re-copy the header and overwrite `.auth-token`. Never commit credentials; `.auth-token` is gitignored.

## Commands

| Command | Method | Endpoint |
| --- | --- | --- |
| `create`     | `POST`   | `/admin/subtenants` |
| `add-user`   | `POST`   | `/admin/subtenants/{id}/users` |
| `list-users` | `GET`    | `/admin/subtenants/{id}/users` |
| `del-user`   | `DELETE` | `/admin/subtenants/{id}/users/{user_id}` |
| `list`       | `GET`    | `/admin/subtenants` |
| `disable`    | `POST`   | `/admin/subtenants/{id}/disable` |
| `enable`     | `POST`   | `/admin/subtenants/{id}/enable` |
| `delete`     | `DELETE` | `/admin/subtenants/{id}` |

Run `./subtenant --help` for the built-in usage text.

## Create a subtenant org

`--subtenant-id` and `--domain` are the only required flags. `--subtenant-id` is the subtenant's unique identifier; `--domain` is the subtenant's single corporate domain. **No user is created** by this step.

The subtenant org's allowed email domains resolve to your **parent tenant's allowed domains ∪ this `--domain`**, and the org is created corporate (domain-restricted invites + JIT).

```bash
# Minimal
./subtenant create --subtenant-id acme-corp --domain acme.com

# Full
./subtenant create \
  --subtenant-id acme-corp \
  --domain acme.com \
  --display-name "Acme Corp" \
  --slug acme \
  --phone +15551234567 \
  --address-street "123 Main St" \
  --address-street2 "Suite 400" \
  --city Austin \
  --state TX \
  --zip 78701 \
  --country US
```

### Create flags

| Flag | Required | Maps to | Notes |
| --- | --- | --- | --- |
| `--subtenant-id` | ✅ | `subtenant_id` | Unique identifier for the subtenant. |
| `--domain` | ✅ | `domain` | The subtenant's single corporate domain (e.g. `acme.com`). |
| `--display-name` | | `display_name` | Human-readable org name (defaults to `--subtenant-id`). |
| `--slug` | | `organization_slug` | |
| `--phone` | | `phone` | |
| `--address-street` | | `address_street` | |
| `--address-street2` | | `address_street2` | |
| `--city` | | `address_city` | |
| `--state` | | `address_state` | |
| `--zip` | | `address_zip_code` | |
| `--country` | | `address_country` | |

Empty optional flags are omitted from the request body. The response includes the resolved `email_allowed_domains`.

> Creating the same `subtenant_id` again with a matching `display_name` + `domain` returns the existing org idempotently; any other conflict (different data, or a different subtenant already using the `domain`) returns `409`.

## Add a user to a subtenant

`<subtenant_id>` (positional or `--subtenant-id`) and `--email` are required. The email's domain must be within the subtenant org's allowed domains, else the call fails with `400`.

```bash
# Add the owner as an admin
./subtenant add-user acme-corp --email owner@acme.com --role admin --first-name Ada --last-name Lovelace

# Add a regular member (default role)
./subtenant add-user acme-corp --email analyst@acme.com
```

### Add-user flags

| Flag | Required | Maps to | Notes |
| --- | --- | --- | --- |
| `--email` | ✅ | `email` | Domain must be in the subtenant's allowed domains. |
| `--role` | | `role` | `admin` or `member` (default `member`). `admin` promotes; `member` never demotes an existing admin. |
| `--first-name` | | `first_name` | |
| `--last-name` | | `last_name` | |
| `--phone` | | `phone` | |

Users are created `active` with the EULA pre-accepted, and receive a magic-link invite. Re-adding an existing user is idempotent (the role rule is applied).

## List / remove users

```bash
./subtenant list-users acme-corp
./subtenant del-user  acme-corp member-...        # <subtenant_id> <user_id>
```

## List subtenants

```bash
./subtenant list
./subtenant list --status active
./subtenant list --status provisioning --limit 100
```

| Flag | Notes |
| --- | --- |
| `--status` | One of `provisioning`, `active`, `disabled`, `deleted`, `failed`. |
| `--limit` | Max rows to return. |

`deleted` rows are hidden unless you ask for them with `--status deleted`.

## Disable / enable / delete a subtenant

All take the id positionally or via `--subtenant-id`:

```bash
./subtenant disable acme-corp                 # pause ingest (status -> disabled)
./subtenant enable  acme-corp                 # resume (disabled -> active)
./subtenant delete  --subtenant-id acme-corp  # soft delete (status -> deleted)
```

- `disable` stops telemetry ingest for the subtenant; ingest is rejected while disabled.
- `enable` restores a **disabled** subtenant to active. `deleted` is terminal and cannot be enabled.
- `delete` is a soft delete; it does not free the `subtenant_id`/`domain` for reuse.

## Output & exit codes

- Progress lines (`>>> POST ...`, `<<< HTTP 200`) go to **stderr**; the response body goes to **stdout**, so you can pipe or capture it cleanly.
- Exits non-zero on any non-2xx response, missing credentials (no `LUMOZ_API_KEY` and no/empty `.auth-token`), or missing required flags.
