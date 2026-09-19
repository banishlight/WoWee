#pragma once

namespace wowee::platform {

/// Mounts the server's Data/ at /Data, downloading each file when it is read
/// (see web_data.cpp). Browser build only. False when the server offers no
/// data, in which case the client starts without.
bool mountWebData();

} // namespace wowee::platform
