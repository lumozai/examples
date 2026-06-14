# Subtenant CLI

A small `bash` wrapper around the `/admin/subtenants` API that lets a **tenant org admin** create and manage subtenants from the command line.

A *subtenant* is a customer-of-a-customer: an isolated organization nested under your tenant. Creating one provisions the org and invites its owner by email.

The CLI talks to the Lumoz production API (`https://api.lumoz.ai`) out of the box — no configuration needed.

## Prerequisites

- `bash`, `curl`, and `python3`.
- `jq` is optional — if present, responses are pretty-printed.
- An admin JWT in `.auth-token` (see below).

## Authentication — getting the `.auth-token`

The CLI reads your JWT from a file named `.auth-token`, next to the `subtenant` script. This is your live console session token, copied from the browser — you must be signed in as a **tenant org admin**.

1. Log in to the console at <https://console.lumoz.ai>.
2. Open **DevTools → Network**, then click around or reload so a request appears.
3. Click any API request, find the **`Authorization`** request header (`Bearer eyJ...`), and copy its value **without the `Bearer ` prefix**.
4. Save just the `eyJ...` token to `.auth-token`:

   ```bash
   echo 'eyJhbGciOi...your.jwt.here' > .auth-token
   ```

5. Verify with `./subtenant list` — a `<<< HTTP 200` means you're in. A `401`/`403` means the token is missing, expired, or not an admin token; grab a fresh one.

> JWTs expire — when calls start returning `401`, re-copy the header and overwrite `.auth-token`. Never commit it; it's a live credential (gitignored here).

## Commands

| Command | Method | Endpoint |
| --- | --- | --- |
| `create`  | `POST`   | `/admin/subtenants` |
| `list`    | `GET`    | `/admin/subtenants` |
| `disable` | `POST`   | `/admin/subtenants/{id}/disable` |
| `delete`  | `DELETE` | `/admin/subtenants/{id}` |

Run `./subtenant --help` for the built-in usage text.

## Create a subtenant

`--id` and `--email` are the only required flags. `--id` is the subtenant's unique identifier; `--email` is the owner who will be invited.

```bash
# Minimal
./subtenant create --id acme-corp --email owner@acme.example.com

# Full
./subtenant create \
  --id acme-corp \
  --email owner@acme.example.com \
  --display-name "Acme Corp" \
  --first-name Ada \
  --last-name Lovelace \
  --phone +15551234567 \
  --address-street "123 Main St" \
  --address-street2 "Suite 400" \
  --city Austin \
  --state TX \
  --zip 78701 \
  --country US \
  --domains acme.com,acme.io \
  --slug acme \
  --metadata '{"tier":"pro"}'
```

### Create flags

| Flag | Required | Maps to | Notes |
| --- | --- | --- | --- |
| `--id` | ✅ | `subtenant_id` | Unique identifier for the subtenant. |
| `--email` | ✅ | `email` | Owner's email; receives the invite. |
| `--display-name` | | `display_name` | Human-readable org name. |
| `--first-name` | | `first_name` | Owner's first name. |
| `--last-name` | | `last_name` | Owner's last name. |
| `--phone` | | `phone` | |
| `--address-street` | | `address_street` | |
| `--address-street2` | | `address_street2` | |
| `--city` | | `address_city` | |
| `--state` | | `address_state` | |
| `--zip` | | `address_zip_code` | |
| `--country` | | `address_country` | |
| `--domains` | | `email_allowed_domains` | Comma-separated; split into a list. |
| `--slug` | | `organization_slug` | |
| `--metadata` | | `metadata` | Must be valid JSON. |

Empty optional flags are omitted from the request body.

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

## Disable / delete a subtenant

Both take the id positionally or via `--id`:

```bash
./subtenant disable acme-corp
./subtenant delete  --id acme-corp
```

## Output & exit codes

- Progress lines (`>>> POST ...`, `<<< HTTP 200`) go to **stderr**; the response body goes to **stdout**, so you can pipe or capture it cleanly.
- Exits non-zero on any non-2xx response, a missing/empty `.auth-token`, or missing required flags.
