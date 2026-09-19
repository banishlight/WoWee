#pragma once

namespace wowee::platform {

/// Mounts the server's Data/ at /Data, fetching each file on first read (see
/// web_data.cpp). Browser build only; must run off the main thread. False
/// when the server offers no data, in which case the client starts without.
bool mountWebData();

} // namespace wowee::platform
