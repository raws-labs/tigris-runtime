# Publishing to the ESP Component Registry

`raws-labs/tigris-runtime` on components.espressif.com. The dual-mode root
`CMakeLists.txt` makes this repository the component; `idf_component.yml` is its
manifest.

## Routine releases (automated)

Pushing a `v*` tag triggers `.github/workflows/publish-component.yml`, which
uploads the component at that version. Keep `idf_component.yml`'s `version` field
in step with the tag. Requires the registry auth to be configured once — either
OIDC trust for this repo, or an `IDF_COMPONENT_API_TOKEN` repository secret (see
the workflow comments).

## First 0.6.0 publish (manual, one-time)

The `v0.6.0` tag predates this manifest, and `develop` is already well past
v0.6.0 (the whole Phase-1 runtime), so the released `0.6.0` component must be
published from the tag with the manifest applied on top. `compote` ships with
the IDF component manager (`pip install idf-component-manager` if absent).

```bash
cd tigris-runtime

# 1. Released v0.6.0 code in a scratch worktree
git worktree add /tmp/tigris-runtime-v060 v0.6.0

# 2. Apply this manifest onto it
git show feature/esp-idf-component-manifest:idf_component.yml \
  > /tmp/tigris-runtime-v060/idf_component.yml

# 3. Load the registry token (must be named IDF_COMPONENT_API_TOKEN in the .env)
set -a; source /path/to/.env; set +a

# 4. Validate, then upload
cd /tmp/tigris-runtime-v060
compote component pack   --name tigris-runtime            # dry validation
compote component upload --namespace raws-labs --name tigris-runtime

# 5. Clean up
cd -
git worktree remove /tmp/tigris-runtime-v060
```

Verify at `https://components.espressif.com/components/raws-labs/tigris-runtime`,
then users install it with `idf.py add-dependency "raws-labs/tigris-runtime"`.
